"""

Module for building structured prompts for conversational models based on templating and conversation history.

This module provides utilities to construct prompts that can be consumed by conversational AI models using a templated approach.

Classes:
    PromptBuilder -- A class to build prompts using a template and optional instructions.

Functions:
    _build_chat_history -- A static method to compile chat history into a dictionary.
    _build_chain_history -- A static method to compile chain history into a dictionary.

Typical usage example:
    builder = PromptBuilder(template_string, ["Ensure politeness", "Stick to the topic"])
    prompt = builder.apply(chain_context)


"""
from typing import Any, List, Optional

from jinja2 import Template

from council.contexts import ChatMessageKind, ChainContext, ContextBase


class PromptBuilder:
    """
    A utility class designed to construct chatbot prompts based on provided templates and context histories.
    This class utilizes a user-defined template to format prompts that incorporate both chat and chain history. The PromptBuilder is flexible in that it allows optional addition of instructions to the resulting prompts.
    
    Attributes:
        _template (Template):
             A string Template object used to render the prompt.
        _instructions (Optional[str]):
             Additional text to be included in the prompt if instructions are provided.
    
    Methods:
        __init__:
             Constructor for the PromptBuilder class.
        apply:
             Generates a formatted prompt based on the current context and additional keyword arguments.
        _build_chat_history:
             A static method that constructs a dictionary of the chat history.
        _build_chain_history:
             A static method that constructs a dictionary of the chain history.
    
    Args:
        t (str):
             The template string to be used for prompt generation.
        instructions (Optional[List[str]]):
             A list of string instructions to be included in the prompt. Defaults to None.
    
    Returns:
        (str):
             A fully rendered prompt string.

    """

    def __init__(self, t: str, instructions: Optional[List[str]] = None):
        """
        Initializes the class with a template and optional instructions.
        
        Args:
            t (str):
                 The string representing the template that is converted into a `Template` object.
            instructions (Optional[List[str]]):
                 A list of instruction strings. If provided and non-empty, they are concatenated into a single string, each prepended with '# INSTRUCTIONS', and appended with a newline character. Otherwise, an empty string is set.
            

        """

        self._template = Template(t)
        if instructions is not None and len(instructions) > 0:
            self._instructions = "\n".join(["# INSTRUCTIONS"] + instructions) + "\n"
        else:
            self._instructions = ""

    def apply(self, context: ChainContext, **kwargs: Any) -> str:
        """
        
        Returns the generated prompt string based on the given context and additional keyword arguments.
            This function constructs a template context by compiling chat and chain histories, incorporating
            pre-set instructions, and merging any additional keyword arguments provided. It then renders this
            context using an internal template to produce a prompt string.
        
        Args:
            context (ChainContext):
                 An object representing the current state and history of interactions
                within the current execution chain.
            **kwargs:
                 Arbitrary keyword arguments that will be merged with the template context.
        
        Returns:
            (str):
                 The rendered prompt string based on the template and the constructed context.

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
    def _build_chat_history(context: ContextBase) -> dict[str, Any]:
        """
        Builds and returns a dictionary representation of the chat history.
        This static method constructs a structured history of the chat by categorizing messages by agent and user, and also provides the last message in the conversation regardless of the sender. It encapsulates the list of messages and last messages from both agent and user into a single dictionary object.
        
        Args:
            context (ContextBase):
                 An object that contains the chat history along with various attributes and methods to interact with the chat context.
        
        Returns:
            (dict[str, Any]):
                 A dictionary containing structured chat history information. This includes separate sub-dictionaries for the agent and user, each containing their respective messages and the last message sent. Also, the last message in the chat history is provided outside of these sub-dictionaries for direct access.
            (The 'agent' sub-dictionary contains the following keys):
            'messages' (list[str]):
                 A list of all the messages sent by the agent.
            'last_message' (str):
                 The last message sent by the agent, or an empty string if there is none.
            (The 'user' sub-dictionary contains the following keys):
            'messages' (list[str]):
                 A list of all the messages sent by the user.
            'last_message' (str):
                 The last message sent by the user, or an empty string if there is none.
            (The top level of the dictionary contains the following keys):
            'messages' (list[ChatMessage]):
                 A list of all the messages in the chat history.
            'last_message' (str):
                 The last message sent in the chat, or an empty string if there is none.
            

        """
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
        """
        Builds a history dictionary based on the ChainContext provided.
        This static method constructs a history of messages from the given ChainContext. It includes all messages in the
        'messages' list and stores the last message separately for quick access. If there are no messages present
        (indicated by 'iteration_count' being zero), both 'messages' and 'last_message' are returned as empty.
        Otherwise, the 'messages' list is populated with the 'message' attribute of each message in the
        'context.current.messages', and 'last_message' holds the content of the last message or an empty string if
        it does not exist.
        
        Args:
            context (ChainContext):
                 The context from which the message history will be built.
        
        Returns:
            (dict[str, Any]):
                 A dictionary containing the 'messages' list and the 'last_message'. The 'messages' list
                includes the content of each message in the chain's context, and 'last_message' is the content of the
                most recent message or an empty string if no last message exists.
            

        """
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
