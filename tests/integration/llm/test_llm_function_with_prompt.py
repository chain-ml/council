import unittest

import dotenv

from council import AzureLLM
from council.llm import LLMFunctionWithPrompt
from council.prompt import LLMPromptConfigObject
from tests import get_data_filename
from tests.integration.llm.test_llm_function import SQLResult
from tests.unit import LLMPrompts

DATASET_DESCRIPTION = """
# DATASET - nyc_airbnb
## Dataset Description
The dataset is the New York City Airbnb Open Data which includes information on Airbnb listings in NYC for the year 2019.
It provides data such as host id and name, geographical coordinates, room types, pricing, etc.

## Tables
### Table Name: NYC_2019

#### Table Description
Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. 
This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.

#### Columns
For each column, the name, data type and description are given as follow : {name}: {data type}: {description}
id: BIGINT: listing ID
name: TEXT: name of the listing
neighbourhood_group: TEXT: location
neighbourhood: TEXT: area
price: BIGINT: price in dollars
"""


class TestLlmFunctionWithPrompt(unittest.TestCase):
    """requires an Azure LLM model deployed"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = AzureLLM.from_env()

        self.prompt_config_simple = LLMPromptConfigObject.from_yaml(get_data_filename(LLMPrompts.sql))
        self.prompt_config_template = LLMPromptConfigObject.from_yaml(get_data_filename(LLMPrompts.sql_template))

    def test_simple_prompt(self):
        llm_func = LLMFunctionWithPrompt(self.llm, SQLResult.from_response, self.prompt_config_simple)
        sql_result = llm_func.execute()
        self.assertIsInstance(sql_result, SQLResult)
        print("", sql_result, sep="\n")

    def test_formatted_prompt(self):
        llm_func = LLMFunctionWithPrompt(
            self.llm,
            SQLResult.from_response,
            self.prompt_config_template,
            system_prompt_params={"dataset_description": DATASET_DESCRIPTION},
        )
        sql_result = llm_func.execute(user_prompt_params={"question": "Show me first 5 rows of the dataset"})
        self.assertIsInstance(sql_result, SQLResult)
        print("", sql_result, sep="\n")
