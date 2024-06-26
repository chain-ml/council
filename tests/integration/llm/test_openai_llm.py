import unittest

import dotenv

from council.contexts import LLMContext
from council.llm import OpenAILLM, LLMMessage, LLMTokenLimitException
from tests import get_data_filename


class TestLlmOpenAI(unittest.TestCase):
    """requires an OpenAI access key"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = OpenAILLM.from_env()

    def test_basic_prompt(self):
        messages = [LLMMessage.user_message("what are the largest cities in South America")]

        result = self.llm.post_chat_request(LLMContext.empty(), messages, model="gpt-3.5-turbo")
        print(result.first_choice)
        messages.append(LLMMessage.system_message(result.first_choice))

        messages = [LLMMessage.user_message("what are the largest cities in South America")]
        result = self.llm.post_chat_request(LLMContext.empty(), messages, model="gpt-4")
        print(result.first_choice)
        messages.append(LLMMessage.system_message(result.first_choice))
        messages.append(LLMMessage.user_message("give me another example"))
        result = self.llm.post_chat_request(LLMContext.empty(), messages, model="gpt-4")
        print(result.first_choice)

    def test_prompt_exceed_token_limit(self):
        messages = [LLMMessage.user_message("what are the largest cities in South America")] * 500
        with self.assertRaises(LLMTokenLimitException) as cm:
            _ = self.llm.post_chat_request(LLMContext.empty(), messages)
        print(str(cm.exception))

    def test_with_image_url(self):
        message = LLMMessage.user_message("What’s in this image?")
        message.add_content(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        )
        messages = [message]
        result = self.llm.post_chat_request(LLMContext.empty(), messages, model="gpt-4o")
        print(result.first_choice)

    def test_with_jpg_image(self):
        message = LLMMessage.user_message("What’s in this image?")
        message.add_content(path=get_data_filename("Gfp-wisconsin-madison-the-nature-boardwalk.jpg"))
        messages = [message]
        result = self.llm.post_chat_request(LLMContext.empty(), messages, model="gpt-4o")
        print(result.first_choice)

    def test_with_png_image(self):
        message = LLMMessage.user_message("What’s in this image?")
        message.add_content(path=get_data_filename("Gfp-wisconsin-madison-the-nature-boardwalk.png"))
        messages = [message]
        result = self.llm.post_chat_request(LLMContext.empty(), messages, model="gpt-4o")
        print(result.first_choice)
