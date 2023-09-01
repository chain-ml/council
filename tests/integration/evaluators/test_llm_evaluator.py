import unittest

import dotenv

from council.contexts import AgentContext, ChatMessage, ChainContext
from council.llm import AzureLLM
from council.evaluators import LLMEvaluator
from council.mocks import MockMonitored
from council.runners import Budget


class TestLlmEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        llm = AzureLLM.from_env()
        self.llm = llm

    def test_basic_prompt_multiple_responses(self):
        evaluator = LLMEvaluator(llm=self.llm)
        context = AgentContext.from_user_message("Hello, who are you?")
        context.new_iteration()

        empty_message = ""
        agent_message = """
            I am an agent!
            How can I help you today?
            """
        rose_message = "Roses are red"

        messages = [empty_message, agent_message, rose_message]
        for index, message in enumerate(messages):
            chain_name = f"chain {index}"
            chain_context = ChainContext.from_agent_context(context, MockMonitored(), chain_name, Budget(10))
            chain_context.append(ChatMessage.skill(message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))

        self.assertGreater(result[1], result[0])
        self.assertGreater(result[1], result[2])

    def test_basic_prompt_multiple_math_responses(self):
        evaluator = LLMEvaluator(llm=self.llm)
        context = AgentContext.from_user_message("Given x=`(8*9)+3`. What is the value of x?")
        context.new_iteration()

        messages = ["x=70", "x=71", "x=75", "x=72", "x=73", "x=78"]
        for index, message in enumerate(messages):
            chain_name = f"chain {index}"
            chain_context = ChainContext.from_agent_context(context, MockMonitored(), chain_name, Budget(10))
            chain_context.append(ChatMessage.skill(message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))

        for i in [0, 1, 3, 4, 5]:
            self.assertGreater(result[2], result[i])

    def test_basic_prompt_single_response(self):
        evaluator = LLMEvaluator(llm=self.llm)
        context = AgentContext.from_user_message("Hello, who are you?")
        context.new_iteration()

        agent_message = """
            I am an agent!
            How can I help you today?
            """

        chain_context = ChainContext.from_agent_context(context, MockMonitored(), "test", Budget(10))
        chain_context.append(ChatMessage.skill(agent_message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))

        self.assertGreater(result[0].score, 5.0)

    def test_basic_prompt_empty_responses(self):
        evaluator = LLMEvaluator(llm=self.llm)
        context = AgentContext.from_user_message("Hello, who are you?")
        context.new_iteration()

        messages = ["", "", "", ""]
        for index, message in enumerate(messages):
            chain_name = f"chain {index}"
            chain_context = ChainContext.from_agent_context(context, MockMonitored(), chain_name, Budget(10))
            chain_context.append(ChatMessage.skill(message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))
        for i in range(0, 3):
            self.assertLessEqual(result[i].score, 2)

    def test_basic_prompt_same_responses(self):
        evaluator = LLMEvaluator(llm=self.llm)
        context = AgentContext.from_user_message("Hello, who are you?")
        context.new_iteration()

        messages = ["I am council", "I am council", "I am council", "I am council"]
        for index, message in enumerate(messages):
            chain_name = f"chain {index}"
            chain_context = ChainContext.from_agent_context(context, MockMonitored(), chain_name, Budget(10))
            chain_context.append(ChatMessage.skill(message, None, "a skill"))

        result = evaluator.execute(context, Budget(10))
        first_score = result[0].score
        for i in range(1, 3):
            self.assertEqual(result[i].score, first_score)
