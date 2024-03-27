"""

A module that provides a client wrapper for handling chat requests with Anthropic's language model APIs.

This module defines a class `AnthropicMessagesLLM` that implements the abstract base class `AnthropicAPIClientWrapper`. The implementation is tailored to interact with the Anthropic language model through the provided API client.

Classes:
    AnthropicMessagesLLM: A client wrapper for sending chat requests and receiving responses from an Anthropic language model.

Functions:
    __init__(self, config: AnthropicLLMConfiguration, client: Anthropic) -> None
        Initializes the `AnthropicMessagesLLM` instance with the necessary configuration and API client.

    post_chat_request(self, messages: Sequence[LLMMessage]) -> List[str]
        Sends a sequence of messages to the Anthropic language model and receives a list of response strings.

    _to_anthropic_messages(messages: Sequence[LLMMessage]) -> Iterable[MessageParam]
        Converts a sequence of LLMMessage instances into a format suitable for the Anthropic language model API.



"""
from __future__ import annotations

from typing import Sequence, List, Iterable, Literal

from anthropic import Anthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import MessageParam

from council.llm import (
    LLMMessage,
    LLMMessageRole,
    AnthropicLLMConfiguration,
)
from council.llm.anthropic import AnthropicAPIClientWrapper


class AnthropicMessagesLLM(AnthropicAPIClientWrapper):
    """
    A wrapper class to handle communication with Anthropic's API client by sending messages and formatting them appropriately.
    This class serves as a mediator between the user's message sequences and the Anthropic API client, ensuring messages are correctly
    formatted and the responses from the API client are properly processed. It uses the configuration settings provided in the
    AnthropicLLMConfiguration object for the API calls.
    
    Attributes:
        _config (AnthropicLLMConfiguration):
             Represents the configuration for the Large Language Model (LLM), such as model
            type, max tokens, timeout, temperature, top_k, and top_p.
        _client (Anthropic):
             The API client instance used to interact with Anthropic's services.
    
    Methods:
        __init__(config:
             AnthropicLLMConfiguration, client: Anthropic) -> None
            Initializes the wrapper with a given configuration and client instance.
        post_chat_request(messages:
             Sequence[LLMMessage]) -> List[str]
            Sends a sequence of messages to the Anthropic API and returns the processed responses as a list of strings.
        _to_anthropic_messages(messages:
             Sequence[LLMMessage]) -> Iterable[MessageParam]
            Converts a sequence of LLMMessage objects into the format expected by the Anthropic API. It also handles the assignment
            of roles ('user' or 'assistant') for each message in the sequence. This is a static method.

    """

    def __init__(self, config: AnthropicLLMConfiguration, client: Anthropic) -> None:
        """
        Initialize the object with the given configuration and client.
        
        Args:
            config (AnthropicLLMConfiguration):
                 An object containing the configuration settings for the Anthropic Language Model.
            client (Anthropic):
                 An instance of the Anthropic client, which facilitates interactions with the language model API.
        
        Returns:
            (None):
                 This method does not return anything as it is the constructor for the class.

        """
        self._config = config
        self._client = client

    def post_chat_request(self, messages: Sequence[LLMMessage]) -> List[str]:
        """
        Sends a chat request with the provided messages to a specific language model and returns the responses.
        This method converts input messages into a format compatible with the chosen language model, sends them to the model for generation,
        and retrieves the generated messages as a list of strings.
        
        Args:
            messages (Sequence[LLMMessage]):
                 A sequence of messages structured in LLMMessage format to be sent to the language model.
        
        Returns:
            (List[str]):
                 A list of strings containing the generated responses from the language model based on the input messages.
        
        Raises:
            Exception:
                 If any issue occurs while sending the requests or processing the responses from the language model.
            

        """
        messages_formatted = self._to_anthropic_messages(messages)
        completion = self._client.messages.create(
            messages=messages_formatted,
            model=self._config.model.unwrap(),
            max_tokens=self._config.max_tokens.unwrap(),
            timeout=self._config.timeout.value,
            temperature=self._config.temperature.unwrap_or(NOT_GIVEN),
            top_k=self._config.top_k.unwrap_or(NOT_GIVEN),
            top_p=self._config.top_p.unwrap_or(NOT_GIVEN),
        )
        return [content.text for content in completion.content]

    @staticmethod
    def _to_anthropic_messages(messages: Sequence[LLMMessage]) -> Iterable[MessageParam]:
        """
        Converts a sequence of LLMMessage objects into an iterable of MessageParam objects, intended to represent messages as either from the user or the assistant, based on the alternating roles and message content aggregation for system messages.
        This method takes a sequence of LLMMessage objects, aggregates the content from system messages, and packages them with non-system messages into MessageParam objects with the roles alternating between 'user' and 'assistant'. The roles alternate after each non-system message, starting with 'user'. The aggregation of system message content continues until a non-system message is encountered, at which point the combined content is assigned to a MessageParam object with the current role. If there is content remaining after the last message has been processed, it is added to a final MessageParam with the appropriate role.
        
        Args:
            messages (Sequence[LLMMessage]):
                 A sequence of LLMMessage objects to be converted into MessageParam objects with role annotations.
        
        Returns:
            (Iterable[MessageParam]):
                 An iterable of MessageParam objects with alternating 'user' and 'assistant' roles.

        """
        result: List[MessageParam] = []
        temp_content = ""
        role: Literal["user", "assistant"] = "user"

        for message in messages:
            if message.is_of_role(LLMMessageRole.System):
                temp_content += message.content
            else:
                temp_content += message.content
                result.append(MessageParam(role=role, content=temp_content))
                temp_content = ""
                role = "assistant" if role == "user" else "user"

        if temp_content:
            result.append(MessageParam(role=role, content=temp_content))

        return result
