import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, FunctionDeclaration, Tool

import pandas as pd
from typing import List, Dict, Tuple
import json
import re

from datafarmer.utils import logger

import asyncio
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_attempt
from tqdm.asyncio import tqdm
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

class Gemini:
    def __init__(
        self,
        project_id: str,
        gemini_version: str= "gemini-1.5-flash",
        generation_config= None,
        safety_settings= None,
        system_instruction= None, 
    ):
        """initializes the gemini wrapper class

        Args:
            data (pd.Dataframe): the dataframe data which contains prompt 
            project_id (str): vertex ai project_id
            gemini_version (str, optional): gemini version. Defaults to "gemini-1.5-flash".
            generation_config (_type_, optional): config of generative model. Defaults to None.
            safety_settings (_type_, optional): safety settings of generative model. Defaults to None.
            system_instruction (_type_, optional): system instruction in the initial generative model. Defaults to None.

        Returns:
            _type_: _description_
        """

        vertexai.init(project=project_id)
        
        self.project_id = project_id
        self.gemini_version = gemini_version
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.system_instruction = system_instruction

        self.generative_model = GenerativeModel(
            model_name = gemini_version,
            system_instruction= system_instruction,
        )
    
    def parse_text(self, text: str, format: str="json") -> Dict:
        """parses the response text from the generative response

        Args:
            response (str): response string
            format (str, optional): the format of the response. Defaults to "json".
        Returns:
            Dict: parsed text
        """
        
        assert format in ["json"], f"Invalid format. currently doesn't support {format} format"

        if format == "json":
            # capture the json markdown code block
            match = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            json_text = match[-1].strip()
            return json.loads(json_text)

    @retry(wait=wait_fixed(60), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
    async def get_async_generation_response(self, prompt: str) -> str:
        """return generation text from the prompt

        Args:
            prompt (str): prompt 

        Returns:
            str: response text
        """
        assert prompt is not None and len(prompt) > 0, "Prompt cannot be empty."

        response = await self.generative_model.generate_content_async(
                [prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                stream=False
            )
            
        return response.text
        
    async def run_async_generation(self) -> List[Tuple[int, str]]:
        """running the list of async generation responses and return the list of responses

        Returns:
            List: list of generation responses in tuple format (index, response)
        """ 

        prompts = self.data["prompt"].tolist()
        tasks = [self.get_async_generation_response(prompt=prompt) for prompt in prompts]
        result = []

        with tqdm(total=len(tasks), desc="Generating", unit="Item") as pbar:
            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    response = await task
                    result.append((i, response))
                except Exception as e:
                    logger.warning(f"Error in processing the item {i+1}, after all the retries")
                finally:
                    pbar.update(1)

        return result
    
    def generate_from_dataframe(self, data: pd.DataFrame, ) -> pd.DataFrame:
        """generate the responses and return the dataframe with the responses

        Args:
            data (pd.DataFrame): dataframe with prompts

        Returns:
            pd.DataFrame: dataframe contains generation result only
        """

        # assertion checks to the data
        logger.info("Starting for generation")

        assert isinstance(data, pd.DataFrame), "data should be a pandas dataframe"
        assert "prompt" in data.columns, "data should have a column named 'prompt'"

        if "id" not in data.columns:
            data = data.reset_index().rename(columns={"index": "id"})
            logger.warning("Data doesn't have 'id' column, so added the index as 'id' column")

        self.data = data

        # get event loop for async
        loop = asyncio.get_event_loop()

        responses = loop.run_until_complete(self.run_async_generation())
        success_rate = len(responses) / len(data)
        logger.info(f"Generation Finished, Success rate: {success_rate:.2%} ({len(responses)}/{len(data)})")

        return pd.DataFrame(responses, columns=["id", "result"])




