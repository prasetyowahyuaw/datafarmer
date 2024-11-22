import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting

import pandas as pd
from typing import List, Tuple, Optional

from datafarmer.utils import logger

import asyncio
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm

import warnings
warnings.filterwarnings("ignore")

class Gemini:
    def __init__(
        self,
        project_id: str,
        gemini_version: str= "gemini-1.5-flash",
        generation_config: Optional[GenerationConfig]= None,
        safety_settings: Optional[SafetySetting] = None,
        system_instruction: Optional[str]= None, 
    ):
        """initialize the Gemini class

        Args:
            project_id (str): google project id
            gemini_version (str, optional): gemini version. Defaults to "gemini-1.5-flash".
            generation_config (Optional[GenerationConfig], optional): generation config. Defaults to None.
            safety_settings (Optional[SafetySetting], optional): gemini generation safety settings. Defaults to None.
            system_instruction (Optional[str], optional): initial instruction in gemini. Defaults to None.
            retry_sleep (int, optional): retry sleep time. Defaults to 90.
        """

        vertexai.init(project=project_id)
        
        self.project_id = project_id
        self.gemini_version = gemini_version
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.system_instruction = system_instruction

        self.generative_model = GenerativeModel(
            model_name= gemini_version,
            generation_config= generation_config,
            safety_settings= safety_settings,
            system_instruction= system_instruction

        )

    @retry(
        wait=wait_fixed(60), 
        stop=stop_after_attempt(2), 
        retry=retry_if_exception_type(Exception),
    )
    async def _get_async_generation_response_with_retry(self, id:str, prompt: str) -> Tuple[str, str]:
        """return generation text from the given prompt with retry

        Args:
            id (str): _description_
            prompt (str): _description_

        Returns:
            Tuple[str, str]: _description_
        """

        response = await self.generative_model.generate_content_async(
                    [prompt],
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    stream=False
                )
    
        return id, response.text, True

    async def _get_async_generation_response(self, id:str, prompt: str) -> Tuple[str, str]:
        """return generation text from the given prompt

        Args:
            id (str): id of the prompt
            prompt (str): prompt text
        Returns:
            str: response text
        """
        assert prompt is not None and len(prompt) > 0, "Prompt cannot be empty."

        try:
            return await self._get_async_generation_response_with_retry(id, prompt)
        except Exception as e:
            logger.warning(f"🚧 All retries failed for id {id}: {str(e)}")
            return id, f"Error: {str(e)}", False
        
    async def _run_async_generation(self, data: pd.DataFrame) -> List[Tuple[str, str]]:
        """running the list of async generation responses and return the list of responses

        Returns:
            List: list of generation responses in tuple format (index, response)
        """ 

        tasks = [
            self._get_async_generation_response(id=row.id, prompt=row.prompt)
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
            logger.warning("🚧 Data doesn't have 'id' column, so added the index as 'id' column")
        
        return data

    async def generate_async_from_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """run the generation async and return the responses in dataframe

        Args:
            data (pd.DataFrame): dataframe with prompts

        Returns:
            pd.DataFrame: dataframe contains generation result only
        """

        data = self._assert_data(data)
        logger.info("🔨 Starting for generation")

        responses = await self._run_async_generation(data)
        success_rate = len(responses) / len(data)
        logger.info(f"✅ Generation Finished, Success rate: {success_rate:.2%} ({len(responses)}/{len(data)})")

        return pd.DataFrame(responses, columns=["id", "result"])

    def generate_from_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """generate the responses from the dataframe

        Args:
            data (pd.DataFrame): dataframe with prompts

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

        return loop.run_until_complete(self.generate_async_from_dataframe(data))


