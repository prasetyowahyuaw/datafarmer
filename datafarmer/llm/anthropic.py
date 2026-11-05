from typing import Optional
from datafarmer.llm.base import BaseLLM, InvalidResponseError
from datafarmer.utils import logger


class Anthropic(BaseLLM):
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        api_key: Optional[str] = None,
        system_instruction: Optional[str] = None,
        max_tokens: int = 8192,
        min_wait: int = 2,
        max_wait: int = 30,
        max_attempts: int = 3,
        request_timeout: int = 30,
        max_concurrency: int = 32,
    ):
        """Initialize the Anthropic class.

        Args:
            model (str, optional): Anthropic model ID. Defaults to "claude-sonnet-5-5".
            api_key (Optional[str], optional): Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            system_instruction (Optional[str], optional): system prompt. Defaults to None.
            max_tokens (int, optional): maximum tokens in response. Defaults to 8192.
            min_wait (int, optional): minimum seconds between retries (exponential backoff). Defaults to 2.
            max_wait (int, optional): maximum seconds between retries. Defaults to 30.
            max_attempts (int, optional): maximum number of retry attempts. Defaults to 3.
            request_timeout (int, optional): per-request timeout in seconds. Defaults to 30.
            max_concurrency (int, optional): maximum simultaneous requests. Defaults to 32.
        """
        super().__init__(min_wait=min_wait, max_wait=max_wait, max_attempts=max_attempts, request_timeout=request_timeout, max_concurrency=max_concurrency)

        try:
            import anthropic as anthropic_sdk
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: uv add anthropic"
            )

        self.model = model
        self.system_instruction = system_instruction
        self.max_tokens = max_tokens
        self.client = anthropic_sdk.AsyncAnthropic(api_key=api_key)

    async def _generate_single(
        self, id: str, prompt: str, **kwargs
    ) -> tuple[str, str, bool]:
        """Generate a single response using the Anthropic API.

        Args:
            id (str): identifier for this prompt
            prompt (str): the prompt text
            **kwargs: unused, kept for interface compatibility

        Returns:
            tuple[str, str, bool]: (id, response_text, is_succeeded)
        """
        create_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.system_instruction:
            create_kwargs["system"] = self.system_instruction

        response = await self.client.messages.create(**create_kwargs, timeout=self.request_timeout)

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text:
            raise InvalidResponseError(
                f"No text content in response (stop_reason={response.stop_reason})"
            )
        if response.stop_reason == "max_tokens":
            logger.warning(
                f"🚧 Response for id {id} was truncated at max_tokens={self.max_tokens}"
            )

        return id, text, True
