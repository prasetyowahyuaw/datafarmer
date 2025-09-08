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

import pandas as pd
from typing import Optional
from itertools import chain
from datafarmer.utils import logger
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_attempt
import asyncio
from tqdm.asyncio import tqdm

import warnings

warnings.filterwarnings("ignore")


class Gemini:
    def __init__(
        self,
        project_id: str,
        google_sdk_version: str = "vertex",
        gemini_version: str = "gemini-1.5-flash",
        generation_config: GenerationConfig | GenerateContentConfig = None,
        safety_settings: Optional[SafetySetting] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[Tool] = None,
    ):
        """initialize the Gemini class

        Args:
            project_id (str): google project id
            google_sdk_version (str, optional): google sdk version, there is vertex and genai. Defaults to "genai".
            gemini_version (str, optional): gemini version. Defaults to "gemini-1.5-flash".
            generation_config (Optional[GenerationConfig], optional): generation config. Defaults to None.
            safety_settings (Optional[SafetySetting], optional): gemini generation safety settings. Defaults to None.
            system_instruction (Optional[str], optional): initial instruction in gemini. Defaults to None.
            tool (Optional[Tool], optional): tool for the generation. Defaults to None.
        """

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
    def _get_binary_file_part(file_path: str, file_type: str = "audio") -> Part:
        """return the Part Class from the given file

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

    @retry(
        wait=wait_fixed(60),
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(Exception),
    )
    async def _get_async_generation_response_with_retry(
        self, id: str, prompt: str, *args, **kwargs
    ) -> tuple[str, str, bool]:
        """return generation text from the given prompt with retry

        Args:
            id (str): the id of the prompt
            prompt (str): prompt that needs to be generated
            *args (Any): additional arguments
            **kwargs (Any): additional keyword arguments, currently the useful ones are `audio_file_path` and `image_file_path`

        Returns:
            tuple[str, str]: it returns the id, generation response and the boolean value if the generation is successful
        """

        contents = [prompt]

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
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    stream=False,
                )

            case "genai":
                response = await self.client.aio.models.generate_content(
                    model=kwargs.get("model", "gemini-2.0-flash"),
                    contents=prompt,
                    config=self.generation_config
                    if kwargs.get("generation_config") is None
                    else kwargs.get("generation_config"),
                )

        return id, response.text, True

    async def _get_async_generation_response(
        self, id: str, prompt: str, *args, **kwargs
    ) -> tuple[str, str, bool]:
        """return generation text from the given prompt

        Args:
            id (str): id of the prompt
            prompt (str): prompt text
        Returns:
            tuple[str, str, bool]: it returns the id, generation response and the boolean value if the generation is successful
        """
        assert prompt is not None and len(prompt) > 0, "Prompt cannot be empty."

        try:
            return await self._get_async_generation_response_with_retry(
                id, prompt, *args, **kwargs
            )
        except Exception as e:
            logger.warning(f"🚧 All retries failed for id {id}: {str(e)}")
            return id, f"Error: {str(e)}", False

    async def _run_async_generation(
        self, data: pd.DataFrame, *args, **kwargs
    ) -> list[tuple[str, str]]:
        """running asynchronously the generation from the dataframe

        Returns:
            List: list of generation responses in tuple (id, result)
        """

        additional_allowed_columns = ["audio_file_path"]

        tasks = [
            self._get_async_generation_response(
                id=row.id,
                prompt=row.prompt,
                *args,
                **{
                    col: getattr(row, col)
                    for col in additional_allowed_columns
                    if col in data.columns
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
        """assert the required attributes of the data"""

        assert isinstance(data, pd.DataFrame), "data should be a pandas dataframe"
        assert "prompt" in data.columns, "data should have a column named 'prompt'"

        if "id" not in data.columns:
            data = data.reset_index().rename(columns={"index": "id"})
            logger.warning(
                "🚧 Data doesn't have 'id' column, so added the index as 'id' column"
            )

        return data

    async def generate_async_from_dataframe(
        self, data: pd.DataFrame, batch_size: int = 120, *args, **kwargs
    ) -> pd.DataFrame:
        """generate asynchronously from dataframe

        Args:
            data (pd.DataFrame): dataframe with prompts
            batch_size (int, optional): batch size for the generation. Defaults to 120.

        Returns:
            pd.DataFrame: dataframe contains id, and result of the generation
        """

        data = self._assert_data(data)
        logger.info("🔨 Starting for generation")

        responses = []
        for i in range(0, len(data), batch_size):
            batch_data = data.iloc[i : i + batch_size]
            logger.info(f"🔄 Processing data batch {i} - {i + len(batch_data)} ...")
            response = await self._run_async_generation(batch_data, *args, **kwargs)
            responses.append(response)

        responses = list(chain(*responses))

        success_rate = len(responses) / len(data)
        logger.info(
            f"✅ Generation Finished, Success rate: {success_rate:.2%} ({len(responses)}/{len(data)})"
        )

        return pd.DataFrame(responses, columns=["id", "result"])

    def generate_from_dataframe(
        self, data: pd.DataFrame, batch_size: int = 120, *args, **kwargs
    ) -> pd.DataFrame:
        """generate from dataframe

        Args:
            data (pd.DataFrame): dataframe with prompts
            batch_size (int, optional): batch size for the generation. Defaults to 120.

        Returns:
            pd.DataFrame: dataframe contains generation result only
        """

        if asyncio.get_event_loop().is_running():
            logger.error("🛑 Use `await generate_async_from_dataframe()` instead")
            raise RuntimeError("Async event loop is already running")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_async_from_dataframe(data, batch_size, *args, **kwargs)
        )
