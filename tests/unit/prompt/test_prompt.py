import unittest

from council.contexts import ChainContext
from council.prompt import PromptBuilder

template = """
User message: {{chat_history.messages[0]}}
Answers:
{%- for message in chat_history.agent.messages %}
{{loop.index}}: {{message}}
{%- endfor %}

The last message is: {{chat_history.last_message}}
"""


class TestPrompt(unittest.TestCase):
    def test(self):
        cc = ChainContext.from_user_message("what are the three largest cities in South America?")
        cc.chatHistory.add_agent_message("São Paulo")
        cc.chatHistory.add_agent_message("Lima")
        cc.chatHistory.add_agent_message("Bogotá")

        p = PromptBuilder(template)
        result = p.apply(cc)

        print(result)
        print("----")

        cc.chatHistory.add_user_message("Thank you")
        result = p.apply(cc)

        print(result)
        print("----")
