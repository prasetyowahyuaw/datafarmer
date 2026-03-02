import subprocess
from typing import Optional
from datafarmer.llm.base import BaseLLM
from datafarmer.utils import logger


class GithubCopilot(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-4o",
        system_instruction: Optional[str] = None,
        github_token: Optional[str] = None,
        min_wait: int = 2,
        max_wait: int = 30,
        max_attempts: int = 3,
        request_timeout: int = 30,
    ):
        """Initialize the GithubCopilot class.

        Uses the GitHub Models API (OpenAI-compatible endpoint at models.inference.ai.azure.com).
        Authentication is via a GitHub personal token, fetched from `gh auth token` if not provided.

        Args:
            model (str, optional): model name available on GitHub Models. Defaults to "gpt-4o".
            system_instruction (Optional[str], optional): system prompt. Defaults to None.
            github_token (Optional[str], optional): GitHub personal token. Falls back to `gh auth token`.
            min_wait (int, optional): minimum seconds between retries (exponential backoff). Defaults to 2.
            max_wait (int, optional): maximum seconds between retries. Defaults to 30.
            max_attempts (int, optional): maximum number of retry attempts. Defaults to 3.
            request_timeout (int, optional): per-request timeout in seconds. Defaults to 30.
        """
        super().__init__(min_wait=min_wait, max_wait=max_wait, max_attempts=max_attempts, request_timeout=request_timeout)

        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: uv add openai"
            )

        self.model = model
        self.system_instruction = system_instruction
        self.token = github_token or self._get_github_token()
        self.client = openai.AsyncOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=self.token,
        )

    @staticmethod
    def _get_github_token() -> str:
        """Fetch the GitHub token via the gh CLI."""
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
        )
        token = result.stdout.strip()
        if not token:
            raise RuntimeError(
                "Could not retrieve GitHub token. Run `gh auth login` first."
            )
        logger.info("🔑 GitHub token retrieved via gh CLI")
        return token

    async def _generate_single(
        self, id: str, prompt: str, **kwargs
    ) -> tuple[str, str, bool]:
        """Generate a single response using the GitHub Models API.

        Args:
            id (str): identifier for this prompt
            prompt (str): the prompt text
            **kwargs: unused, kept for interface compatibility

        Returns:
            tuple[str, str, bool]: (id, response_text, is_succeeded)
        """
        messages = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            timeout=self.request_timeout,
        )
        return id, response.choices[0].message.content, True
