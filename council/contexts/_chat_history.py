"""


Module `_chat_history`
-------------------

This module provides the `ChatHistory` class, which inherits from `MessageList`. It is designed to represent a history of messages within a chat, allowing for message tracking and manipulation throughout the conversation between a user and an agent.

Classes:
    ChatHistory(MessageList): Inherits from `MessageList`. It encapsulates a list of chat messages and offers the same interface with additional static methods specific to chat history functionality.

Static Methods:
    - from_user_message(message: str) -> ChatHistory:
        This static method creates a new instance of `ChatHistory` with a single user message. It is a convenience method for quickly instantiating a chat history beginning with a message from the user.

The `ChatHistory` class can be used to manage a sequential collection of messages and is particularly useful for building and maintaining a record of the conversation between a chatbot (agent) and a user.


"""
from __future__ import annotations
from ._message_list import MessageList


class ChatHistory(MessageList):
    """
    A class that represents the chat history, which is a type of MessageList.
    This class is used to maintain a record of messages exchanged during a chat session.
    It provides a static method to initialize the chat history from a user message.
    
    Attributes:
        Inherits all attributes from the MessageList class.
    
    Methods:
        from_user_message(message:
             str) -> ChatHistory:
            A static method that creates a new instance of ChatHistory
            and adds the initial user message to it.
    
    Args:
        message (str):
             The message from the user to be added to the chat history.
    
    Returns:
        (ChatHistory):
             A new instance of ChatHistory with the initial message added to it.

    """

    @staticmethod
    def from_user_message(message: str) -> ChatHistory:
        """
        Creates a new ChatHistory instance and adds a user message to it.
        This static method initializes a new ChatHistory object, then calls the add_user_message method to add the given message
        to the chat history. It finally returns the updated ChatHistory object with the user message included.
        
        Args:
            message (str):
                 The user message to be added to the chat history.
        
        Returns:
            (ChatHistory):
                 An instance of the ChatHistory class with the user message added to its list of messages.
            

        """
        history = ChatHistory()
        history.add_user_message(message=message)
        return history
