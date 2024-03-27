"""

Module for encapsulating the functionality of requesting completions via the Anthropic API client within a Council environment.

This module defines `AnthropicCompletionLLM`, a class that serves as a wrapper around the Anthropic API client and
implements the `post_chat_request` method as described in the `AnthropicAPIClientWrapper` abstract base class.
The class converts Council messages into a format suitable for the Anthropic API, sends the request, and processes
the response accordingly.

Classes:
    AnthropicCompletionLLM - A class that encapsulates the interaction with the Anthropic API client for generating
                            responses based on a given sequence of LLM messages.

Functions:
    None

Attributes:
    _HUMAN_TURN (str): A constant string representing the human's turn in the conversation as identified by
                     the prompt used for the Anthropic API.
    _ASSISTANT_TURN (str): A constant string representing the assistant's turn in the conversation as
                           identified by the prompt used for the Anthropic API.


"""
from typing import Sequence, List

from anthropic import Anthropic
from anthropic._types import NOT_GIVEN

from council.llm import AnthropicLLMConfiguration, LLMMessage, LLMMessageRole
from council.llm.anthropic import AnthropicAPIClientWrapper

_HUMAN_TURN = Anthropic.HUMAN_PROMPT
_ASSISTANT_TURN = Anthropic.AI_PROMPT


class AnthropicCompletionLLM(AnthropicAPIClientWrapper):
    """
    A wrapper class for Anthropic's API client that facilitates the posting of chat requests to the
    Anthropic language model and processes the responses according to the given configuration.
    
    Attributes:
        _config (AnthropicLLMConfiguration):
             The configuration settings for the language model,
            which include settings like model name, max tokens, timeout, temperature, top_k, and top_p.
        _client (Anthropic):
             The Anthropic API client used to communicate with the language
            model service.
    
    Methods:
        __init__(self, config:
             AnthropicLLMConfiguration, client: Anthropic) -> None:
            Initializes a new instance of the class with the given configuration and API client.
        post_chat_request(self, messages:
             Sequence[LLMMessage]) -> List[str]:
            Sends the chat messages to the Anthropic language model and returns the model's
            response as a list of strings.
        _to_anthropic_messages(messages:
             Sequence[LLMMessage]) -> str:
            Converts a sequence of LLMMessage objects into a format suitable for the Anthropic
            language model by appending role-specific prefixes and joining the messages. This is
            a static method and does not modify instance state.
    
    Raises:
        Exception:
            If the `messages` argument passed to `_to_anthropic_messages` is empty, an exception
            is raised to indicate that no messages are available to process.

    """

    def __init__(self, config: AnthropicLLMConfiguration, client: Anthropic) -> None:
        """
        Initializes a new instance of a class that incorporates the Anthropic language model configuration and client handling.
        
        Args:
            config (AnthropicLLMConfiguration):
                 An instance of the configuration settings specific to the Anthropic language model.
            client (Anthropic):
                 The client instance responsible for interactions with the Anthropic language model services.
        
        Raises:
            TypeError:
                 If the provided arguments are not of the expected types.
        
        Note:
            This method is typically invoked when an instance of the class is created and is responsible for setting up necessary properties or configurations needed for the object to function properly throughout its lifecycle.

        """
        self._config = config
        self._client = client

    def post_chat_request(self, messages: Sequence[LLMMessage]) -> List[str]:
        """
        Sends a chat request to the AI client and returns the response.
        This method takes a sequence of LLMMessage objects, converts them into a
        format understandable by the AI and then sends a request to the AI client.
        The AI's response is captured and returned as a list of strings, typically
        containing the AI's reply to the chat messages.
        
        Args:
            messages (Sequence[LLMMessage]):
                 A sequence of LLMMessage objects
                representing the conversation history or the messages that should
                be sent to the AI.
        
        Returns:
            (List[str]):
                 A list containing the response from the AI client. The list
                consists of a single string which is the AI's reply to the
                provided messages.
        
        Note:
            The configuration for the AI client request such as model, maximum
            tokens to sample, timeout, temperature, top_k, and top_p are taken from
            the '_config' attribute of the instance which should be set prior to
            using this method.
            

        """
        prompt = self._to_anthropic_messages(messages)
        result = self._client.completions.create(
            prompt=prompt,
            model=self._config.model.unwrap(),
            max_tokens_to_sample=self._config.max_tokens.unwrap(),
            timeout=self._config.timeout.value,
            temperature=self._config.temperature.unwrap_or(NOT_GIVEN),
            top_k=self._config.top_k.unwrap_or(NOT_GIVEN),
            top_p=self._config.top_p.unwrap_or(NOT_GIVEN),
        )
        return [result.completion]

    @staticmethod
    def _to_anthropic_messages(messages: Sequence[LLMMessage]) -> str:
        """
        Converts a sequence of LLMMessage objects into a single anthropic-formatted string.
        This method processes a sequence of LLMMessage objects, each potentially
        representing a communication from a user or system, and converts the
        sequence into a readable format where messages are prefixed with identifiers
        to distinguish the roles of the participants. The output is a string where each
        message is on a separate line, beginning with an identifier for the user or
        the assistant based on the message's role. If the first message is from the
        system and there is more than one message, it will be paired with the second
        message prefixed by the user identifier followed by the remaining translated
        messages. When there are no messages provided, an exception is raised indicating
        no messages to process. The function assumes certain global constants defining
        the prefixes for user and assistant turns.
        
        Args:
            messages (Sequence[LLMMessage]):
                 A sequence of LLMMessage objects to be processed.
        
        Returns:
            (str):
                 A string composed of the processed messages, with appropriate role
                prefixes, concatenated together.
        
        Raises:
            Exception:
                 An error if the `messages` sequence is empty.

        """
        messages_count = len(messages)
        if messages_count == 0:
            raise Exception("No message to process.")

        result = []
        if messages[0].is_of_role(LLMMessageRole.System) and messages_count > 1:
            result.append(f"{_HUMAN_TURN} {messages[0].content}\n{messages[1].content}")
            remaining = messages[2:]
        else:
            result.append(f"{_HUMAN_TURN} {messages[0].content}")
            remaining = messages[1:]

        for item in remaining:
            prefix = _HUMAN_TURN if item.is_of_role(LLMMessageRole.User) else _ASSISTANT_TURN
            result.append(f"{prefix} {item.content}")
        result.append(_ASSISTANT_TURN)

        return "".join(result)
