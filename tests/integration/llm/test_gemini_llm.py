import unittest
import dotenv

from council import LLMContext
from council.llm import LLMMessage, GeminiLLM
from council.utils import OsEnviron

from tests import get_data_filename


class TestGeminiLLM(unittest.TestCase):
    def test_completion(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.5-flash"):
            instance = GeminiLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.first_choice

    def test_message(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.0-pro"):
            instance = GeminiLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.first_choice

        messages.append(LLMMessage.user_message("give a famous monument of that place"))
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.5-pro"):
            instance = GeminiLLM.from_env()
            result = instance.post_chat_request(context, messages)

            assert "Eiffel" in result.first_choice

    def test_with_jpg_image(self):
        dotenv.load_dotenv()
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.5-pro"):
            instance = GeminiLLM.from_env()
            message = LLMMessage.user_message("What is in the image?")
            message.add_content(path=get_data_filename("Gfp-wisconsin-madison-the-nature-boardwalk.jpg"))
            result = instance.post_chat_request(LLMContext.empty(), [message])
            print(result.first_choice)

    def test_with_png_image(self):
        dotenv.load_dotenv()
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.5-pro"):
            instance = GeminiLLM.from_env()
            message = LLMMessage.user_message("What is in the image?")
            message.add_content(path=get_data_filename("Gfp-wisconsin-madison-the-nature-boardwalk.png"))
            result = instance.post_chat_request(LLMContext.empty(), [message])
            print(result.first_choice)

    @unittest.skip("Investigate")
    def test_with_image_url(self):
        dotenv.load_dotenv()
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.5-pro"):
            instance = GeminiLLM.from_env()
            message = LLMMessage.user_message("Here is an image.")
            message.add_content(
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            )
            action = LLMMessage.user_message("What is in the image?")
            result = instance.post_chat_request(LLMContext.empty(), [message, action])
            print(result.first_choice)

    def test_consumptions(self):
        messages = [LLMMessage.user_message("Hello how are you?")]
        dotenv.load_dotenv()
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.5-flash-8b"):
            instance = GeminiLLM.from_env()
            result = instance.post_chat_request(LLMContext.empty(), messages)

            assert len(result.consumptions) == 8  # call, duration, 3 token kinds and 3 cost kinds
            for consumption in result.consumptions:
                assert consumption.kind.startswith("gemini-1.5-flash-8b")

    def test_gemini_2_flash_exp(self):
        messages = [LLMMessage.user_message("What is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-2.0-flash-exp"):
            instance = GeminiLLM.from_env()
            result = instance.post_chat_request(LLMContext.empty(), messages)

            assert "Paris" in result.first_choice

            # assert len(result.consumptions) == 8  # call, duration, 3 token kinds and 3 cost kinds
            # for consumption in result.consumptions:
            #     assert consumption.kind.startswith("gemini-2.0-flash-exp")
