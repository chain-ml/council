from __future__ import annotations

import json
import unittest

import dotenv

from council import AzureLLM
from council.llm import LLMParsingException, LLMResponse, LLMMessage
from council.llm.llm_function import LLMFunction
from council.utils import CodeParser

SYSTEM_PROMPT = """
          You are a sql expert solving the `Task` leveraging the database schema in the `DATASET` section.

          # Instructions
          - Assess whether the `Task` is reasonable and possible to solve given the database schema
          - The entire response must be inside a valid `json` code block as defined in the `Response formatting` section
          - Keep your explanation concise with only important details and assumptions, no excuse or other comment

          # Response formatting
 
          Your entire response must be inside the following `json` code block:
          The JSON response schema must contain the following keys: `solved`, `explanation` and `sql`.
          
          ```json
          {
            "solved": {Boolean, indicating whether the task is solved based on the provided database schema},
            "explanation": {String, concise explanation of the solution if solved or reasoning if not solved},
            "sql": {String, the sql query if the task is solved, otherwise empty}
          }
          ```
          
          # DATASET - nyc_airbnb
          ## Dataset Description
          The dataset is the New York City Airbnb Open Data which includes information on Airbnb listings in NYC for the year 2019.
          It provides data such as host id and name, geographical coordinates, room types, pricing, etc.

          ## Tables
          ### Table Name: NYC_2019

          #### Table Description
          Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This table describes the listing activity and metrics in NYC, NY for 2019.
          Content
          This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.

          #### Columns
          For each column, the name, data type and description are given as follow : {name}: {data type}: {description}`
          id: BIGINT: listing ID
          name: TEXT: name of the listing
          neighbourhood_group: TEXT: location
          neighbourhood: TEXT: area
          price: BIGINT: price in dollars
"""

USER = "Price distribution by borough"


class SQLResult:
    def __init__(self, solved: bool, explanation: str, sql: str):
        self.solved = solved
        self.explanation = explanation
        self.sql = sql

    @staticmethod
    def from_response(response: LLMResponse) -> SQLResult:
        llm_response = response.result.first_choice if response.result else ""
        json_block = CodeParser.find_first("json", llm_response)
        if json_block is None:
            raise LLMParsingException("No json block found in response")

        code_response = json.loads(json_block.code)
        sql = code_response.get("sql")
        if sql is not None:
            if "LIMIT" not in sql:
                raise LLMParsingException("Generated SQL query should contain a LIMIT clause")
            return SQLResult(code_response["solved"], code_response["explanation"], sql)
        return SQLResult(False, "No SQL query generated", "")

    def __str__(self):
        if self.solved:
            return f"Sql: {self.sql}\n\nExplanation: {self.explanation}"
        return f"Not solved.\nExplanation: {self.explanation}"


class TestLlmFunction(unittest.TestCase):
    """requires an Azure LLM model deployed"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = AzureLLM.from_env()

    def test_basic_prompt(self):
        llm_func = LLMFunction(self.llm, SQLResult.from_response, SYSTEM_PROMPT)
        sql_result = llm_func.execute(user_message=USER)
        self.assertIsInstance(sql_result, SQLResult)
        print("", sql_result, sep="\n")

    def test_message_prompt(self):
        llm_func = LLMFunction(self.llm, SQLResult.from_response, SYSTEM_PROMPT)
        sql_result = llm_func.execute(user_message=LLMMessage.user_message(USER))
        self.assertIsInstance(sql_result, SQLResult)
        print("", sql_result, sep="\n")

    def test_both_message_prompt_and_messages(self):
        llm_func = LLMFunction(self.llm, SQLResult.from_response, SYSTEM_PROMPT)
        user_message = LLMMessage.user_message(USER)
        messages = [
            LLMMessage.assistant_message("There's not enough information about the dataset to generate SQL"),
            LLMMessage.user_message("Please pay attention to DATASET section"),
        ]
        sql_result = llm_func.execute(user_message=user_message, messages=messages)
        self.assertIsInstance(sql_result, SQLResult)
        print("", sql_result, sep="\n")

    def test_messages_only(self):
        llm_func = LLMFunction(self.llm, SQLResult.from_response, SYSTEM_PROMPT)
        messages = [
            LLMMessage.user_message(USER),
            LLMMessage.assistant_message("There's not enough information about the dataset to generate SQL"),
            LLMMessage.user_message("Please pay attention to DATASET section"),
        ]
        sql_result = llm_func.execute(messages=messages)
        self.assertIsInstance(sql_result, SQLResult)
        print("", sql_result, sep="\n")
