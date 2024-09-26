from typing import Any, Dict, List, Optional

from council.contexts import ChainContext, ChatMessageKind, ContextBase
from jinja2 import Template


class PromptBuilder:
    """
    A class for building prompts using a Jinja2 template and optional instructions.

    Args:
        t (str): The Jinja2 template string for the prompt.
        instructions (Optional[List[str]]): Optional instructions to be appended to the prompt.

    Attributes:
        _template (Template): The Jinja2 template object.
        _instructions (str): The instructions to be appended to the prompt.

    Methods:
        apply(context: ChainContext) -> str:
            Builds and returns the prompt by rendering the template and appending instructions.

    """

    def __init__(self, t: str, instructions: Optional[List[str]] = None) -> None:
        """
        Initializes a PromptBuilder instance.

        Args:
            t (str): The Jinja2 template string for the prompt.
            instructions (Optional[List[str]]): Optional instructions to be appended to the prompt.
        """

        self._template = Template(t)
        if instructions is not None and len(instructions) > 0:
            self._instructions = "\n".join(["# INSTRUCTIONS"] + instructions) + "\n"
        else:
            self._instructions = ""

    def apply(self, context: ChainContext, **kwargs: Any) -> str:
        """
        Builds and returns the prompt by rendering the template and appending instructions.

        Args:
            context (ChainContext): The context object containing the necessary data for rendering the template.

        Returns:
            str: The generated prompt string.

        """

        template_context = {
            "chat_history": self._build_chat_history(context),
            "chain_history": self._build_chain_history(context),
            "instructions": self._instructions,
            **kwargs,
        }

        prompt = self._template.render(template_context)
        return prompt

    @staticmethod
    def _build_chat_history(context: ContextBase) -> Dict[str, Any]:
        last_message = context.chat_history.try_last_message
        last_user_message = context.chat_history.try_last_user_message
        last_agent_message = context.chat_history.try_last_agent_message

        return {
            "agent": {
                "messages": [
                    msg.message for msg in context.chat_history.messages if msg.is_of_kind(ChatMessageKind.Agent)
                ],
                "last_message": last_agent_message.map_or(lambda m: m.message, ""),
            },
            "user": {
                "messages": [
                    msg.message for msg in context.chat_history.messages if msg.is_of_kind(ChatMessageKind.User)
                ],
                "last_message": last_user_message.map_or(lambda m: m.message, ""),
            },
            "messages": [msg for msg in context.chat_history.messages],
            "last_message": last_message.map_or(lambda m: m.message, ""),
        }

    @staticmethod
    def _build_chain_history(context: ChainContext) -> dict[str, Any]:
        if context.iteration_count == 0:
            return {
                "messages": [],
                "last_message": "",
            }

        last_message = context.current.try_last_message
        return {
            "messages": [msg.message for msg in context.current.messages],
            "last_message": last_message.map_or(lambda m: m.message, ""),
        }
