import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    SafetySetting,
    Tool,
    Part,
)
from google import genai
from google.genai.types import GenerateContentConfig

from typing import Optional
from datafarmer.utils import logger
from datafarmer.llm.base import BaseLLM
import json

import warnings

warnings.filterwarnings("ignore")


class Gemini(BaseLLM):
    def __init__(
        self,
        project_id: str,
        google_sdk_version: str = "vertex",
        gemini_version: str = "gemini-2.5-flash-lite",
        generation_config: GenerationConfig | GenerateContentConfig = None,
        safety_settings: Optional[SafetySetting] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[Tool] = None,
        min_wait: int = 10,
        max_wait: int = 60,
        max_attempts: int = 3,
        request_timeout: int = 60,
    ):
        """Initialize the Gemini class.

        Args:
            project_id (str): google project id
            google_sdk_version (str, optional): google sdk version, there is vertex and genai. Defaults to "vertex".
            gemini_version (str, optional): gemini version. Defaults to "gemini-2.5-flash-lite".
            generation_config (Optional[GenerationConfig], optional): generation config. Defaults to None.
            safety_settings (Optional[SafetySetting], optional): gemini generation safety settings. Defaults to None.
            system_instruction (Optional[str], optional): initial instruction in gemini. Defaults to None.
            tools (Optional[Tool], optional): tool for the generation. Defaults to None.
            min_wait (int, optional): minimum seconds between retries (exponential backoff). Defaults to 10.
            max_wait (int, optional): maximum seconds between retries. Defaults to 60.
            max_attempts (int, optional): maximum number of retry attempts. Defaults to 3.
            request_timeout (int, optional): per-request timeout in seconds. Defaults to 60.
        """
        super().__init__(min_wait=min_wait, max_wait=max_wait, max_attempts=max_attempts, request_timeout=request_timeout)

        assert google_sdk_version in [
            "genai",
            "vertex",
        ], "google_sdk_version should be either 'genai' or 'vertex'"

        self.gemini_version = gemini_version
        self.google_sdk_version = google_sdk_version
        self.project_id = project_id
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.system_instruction = system_instruction
        self.tools = tools

        match self.google_sdk_version:
            case "vertex":
                vertexai.init(project=project_id)

                self.generative_model = GenerativeModel(
                    model_name=self.gemini_version,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    system_instruction=self.system_instruction,
                    tools=self.tools,
                )

            case "genai":
                self.client = genai.Client(
                    vertexai=True, project=self.project_id, location="us-central1"
                )

    @staticmethod
    def _get_binary_file_part(
        file_path: str, file_type: str = "audio", *args, **kwargs
    ) -> Part:
        """Return the Part Class from the given file.

        Args:
            file_path (str): the file path
            file_type (str): file type. Defaults to "audio". It can be either "audio" or "image"
        """
        assert file_type in ["audio", "image"], (
            "type should be either 'audio' or 'image'"
        )

        mime = dict(
            audio="audio/mp3",
            image="image/jpeg",
        )

        with open(file_path, "rb") as f:
            binary = f.read()

        file_part = Part.from_data(
            mime_type=mime.get(file_type),
            data=binary,
        )
        logger.info(f"📦 Loaded {file_type} file from {file_path}")
        return file_part

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
        contents = [prompt]
        generation_config = (
            self.generation_config
            if kwargs.get("generation_config") is None
            else kwargs.get("generation_config")
        )

        for key, value in kwargs.items():
            if key == "audio_file_path":
                audio_part = self._get_binary_file_part(
                    file_path=value, file_type="audio"
                )
                contents.append(audio_part)
            elif key == "image_file_path":
                image_part = self._get_binary_file_part(
                    file_path=value, file_type="image"
                )
                contents.append(image_part)

        match self.google_sdk_version:
            case "vertex":
                response = await self.generative_model.generate_content_async(
                    contents,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings,
                    stream=False,
                )

            case "genai":
                response = await self.client.aio.models.generate_content(
                    model=kwargs.get("model", self.gemini_version),
                    contents=prompt,
                    config=generation_config,
                )

        response = response.text

        is_json_response = (
            isinstance(generation_config, GenerateContentConfig)
            and generation_config.response_mime_type == "application/json"
        ) or (
            isinstance(generation_config, dict)
            and generation_config.get("response_mime_type") == "application/json"
        )

        if is_json_response:
            try:
                json.loads(response)
            except Exception:
                raise ValueError(f"Failed to parse JSON response, Id: {id}")

        return id, response, True
