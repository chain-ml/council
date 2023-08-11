import unittest

import dotenv

from council.contexts import ChatHistory, AgentContext, ChatMessage
from council.llm import AzureLLM
from council.evaluators import LLMEvaluator
from council.runners import Budget


class TestLlmEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        llm = AzureLLM.from_env()
        self.llm = llm

    def test_basic_prompt_multiple_responses(self):
        evaluator = LLMEvaluator(llm=self.llm)
        chat_history = ChatHistory.from_user_message(message="Hello, who are you?")
        context = AgentContext(chat_history=chat_history)

        rose_message = "Roses are red"
        empty_message = ""
        agent_message = """
            I am an agent!
            How can I help you today?
            """

        messages = [empty_message, agent_message, rose_message]
        for index, message in enumerate(messages):
            chain_name = f"chain {index}"
            chain_context = context.new_chain_context(chain_name)
            chain_context.current.append(ChatMessage.skill(message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))

        self.assertGreater(result[1], result[0])
        self.assertGreater(result[1], result[2])

    def test_basic_prompt_multiple_math_responses(self):
        evaluator = LLMEvaluator(llm=self.llm)
        chat_history = ChatHistory.from_user_message(message="Could you calculate the value of (8*9)+3")
        context = AgentContext(chat_history=chat_history)

        messages = ["70", "71", "75", "72", "73", "74"]
        for index, message in enumerate(messages):
            chain_name = f"chain {index}"
            chain_context = context.new_chain_context(chain_name)
            chain_context.current.append(ChatMessage.skill(message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))

        for i in [0, 1, 3, 4, 5]:
            self.assertGreater(result[2], result[i])

    def test_basic_prompt_single_response(self):
        evaluator = LLMEvaluator(llm=self.llm)
        chat_history = ChatHistory.from_user_message(message="Hello, who are you?")
        context = AgentContext(chat_history=chat_history)

        agent_message = """
            I am an agent!
            How can I help you today?
            """

        chain_context = context.new_chain_context("test")
        chain_context.current.append(ChatMessage.skill(agent_message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))

        self.assertGreater(result[0].score, 5)

    def test_basic_prompt_empty_responses(self):
        evaluator = LLMEvaluator(llm=self.llm)
        chat_history = ChatHistory.from_user_message(message="Hello, who are you?")
        context = AgentContext(chat_history=chat_history)

        messages = ["", "", "", ""]
        for index, message in enumerate(messages):
            chain_name = f"chain {index}"
            chain_context = context.new_chain_context(chain_name)
            chain_context.current.append(ChatMessage.skill(message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))
        for i in range(0, 3):
            self.assertLessEqual(result[i].score, 2)

    def test_basic_prompt_same_responses(self):
        evaluator = LLMEvaluator(llm=self.llm)
        chat_history = ChatHistory.from_user_message(message="Hello, who are you?")
        context = AgentContext(chat_history=chat_history)

        messages = ["I am council", "I am council", "I am council", "I am council"]
        for index, message in enumerate(messages):
            chain_name = f"chain {index}"
            chain_context = context.new_chain_context(chain_name)
            chain_context.current.append(ChatMessage.skill(message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))
        first_score = result[0].score
        for i in range(1, 3):
            self.assertEqual(result[i].score, first_score)
