import unittest

import dotenv

from council import AzureLLM
from council.llm import LLMParsingException, LLMMessage
from council.llm.llm_function import LLMFunction
from council.llm.llm_response_parser import CodeBlocksResponseParser
from council.prompt import LLMPromptConfigObject
from tests import get_data_filename
from tests.unit import LLMPrompts

prompt_config = LLMPromptConfigObject.from_yaml(get_data_filename(LLMPrompts.sql))
SYSTEM_PROMPT = prompt_config.get_system_prompt_template("default")
USER = prompt_config.get_user_prompt_template("default")


class SQLResult(CodeBlocksResponseParser):
    solved: bool
    explanation: str
    sql: str

    def validator(self) -> None:
        if "limit" not in self.sql.lower():
            raise LLMParsingException("Generated SQL query should contain a LIMIT clause")

    def __str__(self):
        if self.solved:
            return f"SQL: {self.sql}\n\nExplanation: {self.explanation}"
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
