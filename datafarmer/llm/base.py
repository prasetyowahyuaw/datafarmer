from abc import ABC, abstractmethod
from itertools import chain
import asyncio
import pandas as pd
from tenacity import AsyncRetrying, retry_if_exception, wait_exponential, stop_after_attempt
from tqdm.asyncio import tqdm
from datafarmer.utils import logger


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True only for errors that are worth retrying (rate limits, server errors, timeouts)."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    # openai, anthropic, httpx all expose status_code on API errors
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code == 429 or status_code >= 500
    # Fallback: match common retryable class names
    name = type(exc).__name__
    return any(p in name for p in ("Timeout", "Connection", "RateLimit", "ServiceUnavailable", "InternalServer"))


class BaseLLM(ABC):
    def __init__(self, min_wait: int = 2, max_wait: int = 60, max_attempts: int = 3, request_timeout: int = 30):
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout

    @abstractmethod
    async def _generate_single(self, id: str, prompt: str, **kwargs) -> tuple[str, str, bool]:
        """Generate a single response for the given prompt.

        Args:
            id (str): identifier for this prompt
            prompt (str): the prompt text
            **kwargs: provider-specific keyword arguments

        Returns:
            tuple[str, str, bool]: (id, response_text, is_succeeded)
        """
        ...

    async def _get_async_generation_response(
        self, id: str, prompt: str, **kwargs
    ) -> tuple[str, str, bool]:
        """Wrap _generate_single with retry logic.

        Args:
            id (str): identifier for this prompt
            prompt (str): the prompt text
            **kwargs: passed through to _generate_single

        Returns:
            tuple[str, str, bool]: (id, response_text, is_succeeded)
        """
        assert prompt is not None and len(prompt) > 0, "Prompt cannot be empty."

        try:
            async for attempt in AsyncRetrying(
                wait=wait_exponential(multiplier=1, min=self.min_wait, max=self.max_wait),
                stop=stop_after_attempt(self.max_attempts),
                retry=retry_if_exception(_is_retryable_error),
            ):
                with attempt:
                    return await self._generate_single(id, prompt, **kwargs)
        except Exception as e:
            logger.warning(f"🚧 All retries failed for id {id}: {str(e)}")
            return id, f"Error: {str(e)}", False

    async def _run_async_generation(
        self, data: pd.DataFrame, **kwargs
    ) -> list[tuple[str, str]]:
        """Run async generation over a dataframe batch.

        Args:
            data (pd.DataFrame): dataframe with 'id' and 'prompt' columns
            **kwargs: extra kwargs passed to _generate_single

        Returns:
            list[tuple[str, str]]: list of (id, result) pairs for successful generations
        """
        tasks = [
            self._get_async_generation_response(
                id=row.id,
                prompt=row.prompt,
                **{
                    col: getattr(row, col)
                    for col in data.columns
                    if col not in ["id", "prompt"]
                },
                **kwargs,
            )
            for row in data.itertuples()
        ]
        results = []

        with tqdm(total=len(tasks), desc="Generating", unit="Item") as pbar:
            for completed_task in asyncio.as_completed(tasks):
                try:
                    id, response, is_succeeded = await completed_task
                    if is_succeeded:
                        results.append((id, response))
                except Exception as e:
                    logger.error(f"🛑 Error while generating: {str(e)}")
                finally:
                    pbar.update(1)

        return results

    @staticmethod
    def _assert_data(data: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalise the input dataframe."""
        assert isinstance(data, pd.DataFrame), "data should be a pandas dataframe"
        assert "prompt" in data.columns, "data should have a column named 'prompt'"

        if "id" not in data.columns:
            data = data.reset_index().rename(columns={"index": "id"})
            logger.warning(
                "🚧 Data doesn't have 'id' column, so added the index as 'id' column"
            )

        return data

    async def generate_async_from_dataframe(
        self, data: pd.DataFrame, batch_size: int = 120, **kwargs
    ) -> pd.DataFrame:
        """Generate responses asynchronously from a dataframe.

        Args:
            data (pd.DataFrame): dataframe with a 'prompt' column (and optionally 'id')
            batch_size (int): number of rows per batch. Defaults to 120.
            **kwargs: passed through to _generate_single

        Returns:
            pd.DataFrame: dataframe with columns ['id', 'result']
        """
        data = self._assert_data(data)
        logger.info("🔨 Starting for generation")

        responses = []
        for i in range(0, len(data), batch_size):
            batch_data = data.iloc[i : i + batch_size]
            logger.info(f"🔄 Processing data batch {i} - {i + len(batch_data)} ...")
            response = await self._run_async_generation(batch_data, **kwargs)
            responses.append(response)

        responses = list(chain(*responses))

        success_rate = len(responses) / len(data)
        logger.info(
            f"✅ Generation Finished, Success rate: {success_rate:.2%} ({len(responses)}/{len(data)})"
        )

        return pd.DataFrame(responses, columns=["id", "result"])

    def generate_from_dataframe(
        self, data: pd.DataFrame, batch_size: int = 120, **kwargs
    ) -> pd.DataFrame:
        """Synchronous wrapper around generate_async_from_dataframe.

        Args:
            data (pd.DataFrame): dataframe with a 'prompt' column (and optionally 'id')
            batch_size (int): number of rows per batch. Defaults to 120.
            **kwargs: passed through to _generate_single

        Returns:
            pd.DataFrame: dataframe with columns ['id', 'result']
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            logger.error("🛑 Use `await generate_async_from_dataframe()` instead")
            raise RuntimeError("Async event loop is already running")

        return loop.run_until_complete(
            self.generate_async_from_dataframe(data, batch_size, **kwargs)
        )
