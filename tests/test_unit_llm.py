"""Offline unit tests for the LLM layer (no network / credentials required)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from google.genai import types

from datafarmer.llm import Anthropic, Gemini, GithubCopilot, VertexRag
from datafarmer.llm.base import BaseLLM, InvalidResponseError, _is_retryable_error


class FakeLLM(BaseLLM):
    """Records calls; behaviour is driven by a handler(id, prompt, **kw)."""

    def __init__(self, handler, **kw):
        kw.setdefault("min_wait", 0)
        kw.setdefault("max_wait", 0)
        super().__init__(**kw)
        self.handler = handler
        self.calls = []

    async def _generate_single(self, id, prompt, **kwargs):
        self.calls.append((id, prompt, kwargs))
        return id, await self.handler(id, prompt, **kwargs), True


class HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


# ---------------------------------------------------------------- base


def test_generate_from_dataframe_basic_and_default_id():
    async def handler(id, prompt, **kw):
        return prompt.upper()

    llm = FakeLLM(handler)
    out = llm.generate_from_dataframe(pd.DataFrame({"prompt": ["a", "b"]}))
    assert sorted(out["result"]) == ["A", "B"]
    assert sorted(out["id"]) == [0, 1]


def test_extra_columns_forwarded_and_nan_dropped():
    async def handler(id, prompt, **kw):
        return "ok"

    llm = FakeLLM(handler)
    df = pd.DataFrame(
        {"id": [1, 2], "prompt": ["a", "b"], "audio_file_path": ["x.mp3", None]}
    )
    llm.generate_from_dataframe(df)
    by_id = {c[0]: c[2] for c in llm.calls}
    assert by_id[1] == {"audio_file_path": "x.mp3"}
    assert by_id[2] == {}  # NaN cell must not reach the provider


def test_non_identifier_column_names_do_not_break():
    async def handler(id, prompt, **kw):
        return "ok"

    llm = FakeLLM(handler)
    llm.generate_from_dataframe(
        pd.DataFrame({"id": [1], "prompt": ["a"], "my col": ["v"]})
    )
    assert llm.calls[0][2] == {"my col": "v"}


def test_max_concurrency_is_respected():
    state = {"cur": 0, "peak": 0}

    async def handler(id, prompt, **kw):
        state["cur"] += 1
        state["peak"] = max(state["peak"], state["cur"])
        await asyncio.sleep(0.01)
        state["cur"] -= 1
        return "ok"

    llm = FakeLLM(handler, max_concurrency=3)
    llm.generate_from_dataframe(pd.DataFrame({"prompt": ["p"] * 20}))
    assert state["peak"] == 3


def test_retry_on_retryable_then_success():
    attempts = {"n": 0}

    async def handler(id, prompt, **kw):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise HttpError(429)
        return "ok"

    out = FakeLLM(handler, max_attempts=3).generate_from_dataframe(
        pd.DataFrame({"prompt": ["a"]})
    )
    assert attempts["n"] == 3
    assert out["result"].tolist() == ["ok"]


def test_no_retry_on_client_error_and_row_dropped():
    attempts = {"n": 0}

    async def handler(id, prompt, **kw):
        attempts["n"] += 1
        raise HttpError(401)

    out = FakeLLM(handler, max_attempts=3).generate_from_dataframe(
        pd.DataFrame({"prompt": ["a"]})
    )
    assert attempts["n"] == 1
    assert out.empty and list(out.columns) == ["id", "result"]


def test_invalid_response_is_retried_and_final_error_is_not_wrapped(caplog):
    attempts = {"n": 0}

    async def handler(id, prompt, **kw):
        attempts["n"] += 1
        raise InvalidResponseError("bad json")

    FakeLLM(handler, max_attempts=2).generate_from_dataframe(
        pd.DataFrame({"prompt": ["a"]})
    )
    assert attempts["n"] == 2


def test_is_retryable_error():
    assert _is_retryable_error(HttpError(503))
    assert _is_retryable_error(asyncio.TimeoutError())
    assert _is_retryable_error(InvalidResponseError())
    assert not _is_retryable_error(HttpError(400))
    assert not _is_retryable_error(ValueError("x"))


def test_sync_wrapper_raises_inside_running_loop():
    async def handler(id, prompt, **kw):
        return "ok"

    llm = FakeLLM(handler)

    async def inner():
        llm.generate_from_dataframe(pd.DataFrame({"prompt": ["a"]}))

    with pytest.raises(RuntimeError):
        asyncio.run(inner())


def test_async_api_inside_running_loop():
    async def handler(id, prompt, **kw):
        return "ok"

    llm = FakeLLM(handler)
    out = asyncio.run(
        llm.generate_async_from_dataframe(pd.DataFrame({"prompt": ["a"]}))
    )
    assert out["result"].tolist() == ["ok"]


def test_missing_prompt_column_rejected():
    with pytest.raises(AssertionError):
        BaseLLM._assert_data(pd.DataFrame({"x": [1]}))


# ---------------------------------------------------------------- anthropic


def _anthropic(response):
    llm = Anthropic(api_key="test", system_instruction="sys", min_wait=0, max_wait=0)
    llm.client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))
    return llm


def _block(text, type="text"):
    return SimpleNamespace(type=type, text=text)


def test_anthropic_default_model_is_current():
    assert Anthropic(api_key="t").model == "claude-sonnet-5-5"


def test_anthropic_joins_text_blocks_and_skips_thinking():
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="..."), _block("a"), _block("b")],
        stop_reason="end_turn",
    )
    llm = _anthropic(resp)
    out = llm.generate_from_dataframe(pd.DataFrame({"id": [1], "prompt": ["hi"]}))
    assert out["result"].tolist() == ["ab"]
    kwargs = llm.client.messages.create.call_args.kwargs
    assert kwargs["system"] == "sys" and kwargs["timeout"] == llm.request_timeout


def test_anthropic_empty_content_is_failure():
    llm = _anthropic(SimpleNamespace(content=[], stop_reason="end_turn"))
    llm.max_attempts = 1
    out = llm.generate_from_dataframe(pd.DataFrame({"id": [1], "prompt": ["hi"]}))
    assert out.empty


# ---------------------------------------------------------------- github models


def test_github_models_defaults_and_alias():
    from datafarmer.llm.github_copilot import GitHubModels

    assert GitHubModels is GithubCopilot
    llm = GithubCopilot(github_token="t")
    assert llm.model == "openai/gpt-4.1"
    assert str(llm.client.base_url).startswith("https://models.github.ai/inference")


def test_github_token_missing_gh_cli():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="gh"):
            GithubCopilot._get_github_token()


# ---------------------------------------------------------------- gemini


@pytest.fixture
def gemini():
    with patch("datafarmer.llm.gemini.genai.Client") as client_cls:
        g = Gemini(project_id="p", min_wait=0, max_wait=0)
        g.client = client_cls.return_value
        g.client.aio.models.generate_content = AsyncMock(
            return_value=SimpleNamespace(text="hello")
        )
        yield g


def test_gemini_rejects_legacy_vertex_sdk():
    with pytest.raises(ValueError, match="no longer supported"):
        Gemini(project_id="p", google_sdk_version="vertex")


def test_gemini_client_defaults_and_overrides():
    with patch("datafarmer.llm.gemini.genai.Client") as client_cls:
        Gemini(project_id="p", client_kwargs={"location": "global"})
        client_cls.assert_called_once_with(vertexai=True, project="p", location="global")


def test_gemini_config_applies_system_instruction_safety_tools_on_genai():
    with patch("datafarmer.llm.gemini.genai.Client"):
        tool = types.Tool(google_search=types.GoogleSearch())
        g = Gemini(
            project_id="p",
            system_instruction="be brief",
            tools=[tool],
            generation_config={"temperature": 0.1},
        )
    cfg = g._build_config(None)
    assert cfg.system_instruction == "be brief"
    assert cfg.tools == [tool]
    assert cfg.temperature == 0.1
    # per-call override does not mutate the instance default
    cfg2 = g._build_config(types.GenerateContentConfig(temperature=0.9))
    assert cfg2.temperature == 0.9 and g.generation_config == {"temperature": 0.1}


def test_gemini_attaches_audio_and_image_parts(gemini, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    img = tmp_path / "i.png"
    img.write_bytes(b"\x89PNG")
    df = pd.DataFrame(
        {
            "id": [1],
            "prompt": ["describe"],
            "audio_file_path": [str(audio)],
            "image_file_path": [str(img)],
        }
    )
    out = gemini.generate_from_dataframe(df)
    assert out["result"].tolist() == ["hello"]
    contents = gemini.client.aio.models.generate_content.call_args.kwargs["contents"]
    assert contents[0] == "describe" and len(contents) == 3
    assert contents[1].inline_data.mime_type in ("audio/x-wav", "audio/wav")
    assert contents[2].inline_data.mime_type == "image/png"


def test_gemini_per_row_model_and_nan_model_falls_back(gemini):
    df = pd.DataFrame({"id": [1, 2], "prompt": ["a", "b"], "model": ["m-x", None]})
    gemini.generate_from_dataframe(df)
    models = sorted(
        c.kwargs["model"] for c in gemini.client.aio.models.generate_content.call_args_list
    )
    assert models == ["gemini-3.1-flash-lite", "m-x"]


def test_gemini_invalid_json_is_retried_then_dropped(gemini):
    gemini.generation_config = {"response_mime_type": "application/json"}
    gemini.client.aio.models.generate_content.return_value = SimpleNamespace(text="nope")
    gemini.max_attempts = 2
    out = gemini.generate_from_dataframe(pd.DataFrame({"id": [1], "prompt": ["a"]}))
    assert out.empty
    assert gemini.client.aio.models.generate_content.await_count == 2


def test_gemini_valid_json_passes(gemini):
    gemini.generation_config = types.GenerateContentConfig(response_mime_type="application/json")
    gemini.client.aio.models.generate_content.return_value = SimpleNamespace(text='{"a": 1}')
    out = gemini.generate_from_dataframe(pd.DataFrame({"id": [1], "prompt": ["a"]}))
    assert out["result"].tolist() == ['{"a": 1}']


def test_gemini_request_timeout_enforced(gemini):
    async def slow(**kw):
        await asyncio.sleep(1)

    gemini.client.aio.models.generate_content = slow
    gemini.request_timeout = 0.01
    gemini.max_attempts = 1
    out = gemini.generate_from_dataframe(pd.DataFrame({"id": [1], "prompt": ["a"]}))
    assert out.empty


# ---------------------------------------------------------------- vertex rag


@pytest.fixture
def vrag():
    with patch("datafarmer.llm.vertex_rag.vertexai.init"):
        yield VertexRag(project_id="p")


def test_rag_tool_is_genai_tool_and_usable_by_gemini(vrag):
    tool = vrag.get_rag_tool("projects/p/locations/l/ragCorpora/1", 5, 0.4)
    assert isinstance(tool, types.Tool)
    store = tool.retrieval.vertex_rag_store
    assert store.similarity_top_k == 5 and store.vector_distance_threshold == 0.4
    assert store.rag_resources[0].rag_corpus.endswith("/1")
    with patch("datafarmer.llm.gemini.genai.Client"):
        g = Gemini(project_id="p", tools=[tool])
    assert g._build_config(None).tools == [tool]


def test_rag_set_corpus_uses_embedding_model(vrag):
    with patch("datafarmer.llm.vertex_rag.rag.create_corpus") as create:
        vrag.set_corpus("name", embedding_model="gemini-embedding-001")
    cfg = create.call_args.kwargs["backend_config"]
    endpoint = cfg.rag_embedding_model_config.vertex_prediction_endpoint
    assert endpoint.publisher_model == "publishers/google/models/gemini-embedding-001"


def test_rag_import_files_chunking(vrag):
    with patch("datafarmer.llm.vertex_rag.rag.import_files") as imp:
        imp.return_value = MagicMock(imported_rag_files_count=2)
        vrag.import_files_to_rag("corpus", ["gs://b/x"], chunk_size=256, chunk_overlap=10)
    chunking = imp.call_args.kwargs["transformation_config"].chunking_config
    assert (chunking.chunk_size, chunking.chunk_overlap) == (256, 10)


def test_rag_retrieval_query(vrag):
    with patch("datafarmer.llm.vertex_rag.rag.retrieval_query") as q:
        vrag.get_retrieval_query("corpus", "refund?", similarity_top_k=3, vector_distance_threshold=0.2)
    kw = q.call_args.kwargs
    assert kw["text"] == "refund?"
    assert kw["rag_retrieval_config"].top_k == 3
