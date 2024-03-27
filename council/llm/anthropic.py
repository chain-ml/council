"""

A module for interacting with Anthropic API.

This module provides an abstract base class `AnthropicAPIClientWrapper`, which serves as a template for
clients intended to communicate with the Anthropic API. It defines the structure and expected behaviors
of such clients through abstract methods.

Classes:
    AnthropicAPIClientWrapper(ABC): An abstract base class that all API client wrappers must inherit.
    This class defines the essential methods that need to be implemented for communication with the
    Anthropic API.

    The purpose of this class is to establish a clear interface for sending chat requests and
    formalize the expected input and output types to ensure reliable interactions with the API.

Abstract Methods:
    post_chat_request(messages: Sequence[LLMMessage]) -> List[str]: This method is an abstract
    method that should be overridden by the subclasses of `AnthropicAPIClientWrapper`. It is expected
to send a sequence of chat messages to the Anthropic API and return a list of response strings.

Attributes:
    None

Functions:
    None

Note:
    As an ABC, `AnthropicAPIClientWrapper` cannot be instantiated directly and requires concrete
    implementation of its abstract methods in a subclass.


"""
import abc
from abc import ABC
from typing import Sequence, List

from council.llm import LLMMessage


class AnthropicAPIClientWrapper(ABC):

    """
    A base class wrapper for Anthropic API clients to interact with language models.
    This abstract class serves as a contract for subclasses to implement a method to send chat
    requests to an API endpoint provided by Anthropic. Subclasses should handle the
    implementation details required to communicate with the Anthropic language models and
    process the responses.
    
    Attributes:
        None explicitly declared; this class serves as an interface.
    
    Methods:
        post_chat_request(messages):
             An abstract method that subclasses must override to send
            a sequence of messages to the respective API and obtain responses.
    
    Args:
        messages (Sequence[LLMMessage]):
             A sequence of LLMMessage objects that encapsulate
            the conversation history or messages to be sent to the language model.
    
    Returns:
        (List[str]):
             A list of response strings received from the API after processing the
            chat request.

    """
    @abc.abstractmethod
    def post_chat_request(self, messages: Sequence[LLMMessage]) -> List[str]:
        """
        Sends a sequence of chat messages to be processed by a language model and returns their responses.
        This abstract method defines the interface for a function that takes a sequence of LLMMessage objects,
        which encapsulates the messages intended for a chat-based interaction with a language model, and
        is expected to return a list of strings, each corresponding to the language model's response to the
        input messages.
        
        Args:
            messages (Sequence[LLMMessage]):
                 A sequence of message objects that contain the data to be sent
                to a language model for processing.
        
        Returns:
            (List[str]):
                 A list of response strings from the language model, with each response corresponding
                to each input message.

        """
        pass
