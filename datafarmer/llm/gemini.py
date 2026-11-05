import asyncio
import json
import mimetypes
from typing import Optional, Union

from google import genai
from google.genai import types

from datafarmer.llm.base import BaseLLM, InvalidResponseError
from datafarmer.utils import logger

_DEFAULT_MIME = dict(audio="audio/mpeg", image="image/jpeg")


class Gemini(BaseLLM):
    def __init__(
        self,
        project_id: str,
        google_sdk_version: str = "genai",
        gemini_version: str = "gemini-3.1-flash-lite",
        generation_config: Union[types.GenerateContentConfig, dict, None] = None,
        safety_settings: Optional[list[types.SafetySetting]] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[list[types.Tool]] = None,
        min_wait: int = 10,
        max_wait: int = 60,
        max_attempts: int = 3,
        request_timeout: int = 60,
        client_kwargs: Optional[dict] = None,
        max_concurrency: int = 32,
    ):
        """Initialize the Gemini class (backed by the ``google-genai`` SDK on Vertex AI).

        Args:
            project_id (str): google project id
            google_sdk_version (str, optional): kept for backward compatibility. Only "genai" is
                supported; the legacy Vertex AI SDK ("vertex") was removed by Google.
            gemini_version (str, optional): gemini model name. Defaults to "gemini-3.1-flash-lite".
            generation_config (Union[GenerateContentConfig, dict], optional): generation config. Defaults to None.
            safety_settings (list[SafetySetting], optional): gemini safety settings. Defaults to None.
            system_instruction (Optional[str], optional): system instruction. Defaults to None.
            tools (list[Tool], optional): tools for the generation (e.g. from ``VertexRag.get_rag_tool``).
            min_wait (int, optional): minimum seconds between retries (exponential backoff). Defaults to 10.
            max_wait (int, optional): maximum seconds between retries. Defaults to 60.
            max_attempts (int, optional): maximum number of attempts per prompt. Defaults to 3.
            request_timeout (int, optional): per-request timeout in seconds. Defaults to 60.
            client_kwargs (Optional[dict], optional): extra keyword arguments forwarded to
                ``genai.Client`` (e.g. ``location``, ``http_options``, ``credentials``,
                ``api_key``). Keys here override the defaults (``vertexai=True``,
                ``project=project_id``, ``location='us-central1'``). Defaults to None.
            max_concurrency (int, optional): maximum simultaneous requests. Defaults to 32.
        """
        super().__init__(
            min_wait=min_wait,
            max_wait=max_wait,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            max_concurrency=max_concurrency,
        )

        if google_sdk_version != "genai":
            raise ValueError(
                "google_sdk_version='vertex' is no longer supported: the legacy Vertex AI "
                "generative SDK was removed by Google. Use 'genai' (the default)."
            )

        self.gemini_version = gemini_version
        self.google_sdk_version = google_sdk_version
        self.project_id = project_id
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.system_instruction = system_instruction
        self.tools = tools
        self.client_kwargs = client_kwargs or {}

        client_kwargs = {
            "vertexai": True,
            "project": self.project_id,
            "location": "us-central1",
        }
        client_kwargs.update(self.client_kwargs)
        self.client = genai.Client(**client_kwargs)

    @staticmethod
    def _get_binary_file_part(
        file_path: str, file_type: str = "audio", *args, **kwargs
    ) -> types.Part:
        """Return a Part from the given file.

        Args:
            file_path (str): the file path
            file_type (str): file type. Defaults to "audio". It can be either "audio" or "image"
        """
        assert file_type in ["audio", "image"], (
            "type should be either 'audio' or 'image'"
        )

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None or not mime_type.startswith(f"{file_type}/"):
            mime_type = _DEFAULT_MIME[file_type]

        with open(file_path, "rb") as f:
            binary = f.read()

        logger.info(f"📦 Loaded {file_type} file from {file_path}")
        return types.Part.from_bytes(data=binary, mime_type=mime_type)

    def _build_config(
        self, generation_config: Union[types.GenerateContentConfig, dict, None]
    ) -> types.GenerateContentConfig:
        """Merge the per-call generation config with the instance-level defaults."""
        if generation_config is None:
            generation_config = self.generation_config

        if generation_config is None:
            config = types.GenerateContentConfig()
        elif isinstance(generation_config, dict):
            config = types.GenerateContentConfig(**generation_config)
        else:
            config = generation_config.model_copy()

        if self.system_instruction and config.system_instruction is None:
            config.system_instruction = self.system_instruction
        if self.safety_settings and config.safety_settings is None:
            config.safety_settings = self.safety_settings
        if self.tools and config.tools is None:
            config.tools = self.tools

        return config

    async def _generate_single(
        self, id: str, prompt: str, **kwargs
    ) -> tuple[str, str, bool]:
        """Generate a single response using the Gemini API.

        Args:
            id (str): identifier for this prompt
            prompt (str): the prompt text
            **kwargs: supports audio_file_path, image_file_path, generation_config, model

        Returns:
            tuple[str, str, bool]: (id, response_text, is_succeeded)
        """
        config = self._build_config(kwargs.get("generation_config"))
        contents = [prompt]

        if kwargs.get("audio_file_path"):
            contents.append(
                self._get_binary_file_part(kwargs["audio_file_path"], file_type="audio")
            )
        if kwargs.get("image_file_path"):
            contents.append(
                self._get_binary_file_part(kwargs["image_file_path"], file_type="image")
            )

        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=kwargs.get("model") or self.gemini_version,
                contents=contents,
                config=config,
            ),
            timeout=self.request_timeout,
        )

        text = response.text
        if text is None:
            raise InvalidResponseError(f"Empty response from model, Id: {id}")

        if config.response_mime_type == "application/json":
            try:
                json.loads(text)
            except Exception:
                raise InvalidResponseError(f"Failed to parse JSON response, Id: {id}")

        return id, text, True
