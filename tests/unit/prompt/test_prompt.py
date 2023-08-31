import unittest

from council.contexts import ChainContext, ChatHistory
from council.prompt import PromptBuilder

template = """
User message: {{chat_history.messages[0]}}
Answers:
{%- for message in chat_history.agent.messages %}
{{loop.index}}: {{message}}
{%- endfor %}

The last message is: {{chat_history.last_message}}
"""

template_ch = """
{{instructions}}
{%- if chat_history.messages|length >0 %}
# CONVERSATION HISTORY
{%- for message in chat_history.messages %}
- {{message.kind}}: {{message.message}}
{%- endfor %}
{%- endif %}

{{test}}
"""


class TestPrompt(unittest.TestCase):
    def test(self):
        chat_history = ChatHistory.from_user_message("what are the three largest cities in South America?")
        chat_history.add_agent_message("São Paulo")
        chat_history.add_agent_message("Lima")
        chat_history.add_agent_message("Bogotá")

        cc = ChainContext.from_chat_history(chat_history)
        prompt_builder = PromptBuilder(template)
        result = prompt_builder.apply(cc)
        self.assertTrue(result.strip().endswith("Bogotá"))

        print(result)
        print("----")

        cc.chat_history.add_user_message("Thank you")
        result = prompt_builder.apply(cc)
        self.assertTrue(result.strip().endswith("Thank you"))

        print(result)
        print("----")

    def test_conversation_history(self):
        cc = ChainContext.from_user_message("what are the three largest cities in South America?")
        cc.chat_history.add_agent_message("São Paulo")
        cc.chat_history.add_agent_message("Lima")
        cc.chat_history.add_user_message("give another one.")
        cc.chat_history.add_agent_message("Bogotá")

        prompt_builder = PromptBuilder(template_ch)
        result = prompt_builder.apply(cc)

        self.assertIn("what are the three largest cities in South America?", result)
        self.assertTrue(result.strip().endswith("Bogotá"))
        print(result)

    def test_instructions(self):
        cc = ChainContext.from_user_message("what are the three largest cities in South America?")
        prompt_builder = PromptBuilder(template_ch, instructions=["Ensure to give accurate information"])
        result = prompt_builder.apply(cc)

        self.assertIn("Ensure to give accurate information", result)
        print(result)

    def test_parameters(self):
        cc = ChainContext.from_user_message("what are the three largest cities in South America?")
        prompt_builder = PromptBuilder(template_ch)
        result = prompt_builder.apply(cc, test="value")

        self.assertTrue(result.endswith("value"))
        print(result)
