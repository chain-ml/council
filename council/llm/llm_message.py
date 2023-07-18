from enum import Enum
from typing import Optional, List, Iterable

from council.contexts import ChatMessage, ChatMessageKind


class LLMMessageRole(str, Enum):
    """
    Enum representing the roles of messages in a conversation or dialogue.
    """

    User = "user"
    """
    Represents a message from the user.
    """

    System = "system"
    """
    Represents a system-generated message.
    """

    Assistant = "assistant"
    """
    Represents a message from the assistant.
    """


class LLMMessage:
    """
    Represents chat messages. Used in the payload

    Args:
        role (LLMMessageRole): the role/persona the message is coming from. Could be either user, system or assistant
        content (str): the message content
    """

    _role: LLMMessageRole
    _content: str

    def __init__(self, role: LLMMessageRole, content: str):
        """Initialize a new instance"""
        self._role = role
        self._content = content

    @staticmethod
    def system_message(content: str) -> "LLMMessage":
        """
        Create a new system message

        Parameters:
            content (str): the message content
        """
        return LLMMessage(role=LLMMessageRole.System, content=content)

    @staticmethod
    def user_message(content: str) -> "LLMMessage":
        """
        Create a new user message

        Parameters:
            content (str): the message content
        """
        return LLMMessage(role=LLMMessageRole.User, content=content)

    @staticmethod
    def assistant_message(content: str) -> "LLMMessage":
        """
        Create a new assistant message

        Parameters:
            content (str): the message content
        """
        return LLMMessage(role=LLMMessageRole.Assistant, content=content)

    def dict(self) -> dict[str, str]:
        return {"role": self._role.value, "content": self._content}

    @property
    def content(self) -> str:
        """Retrieve the content of this instance"""
        return self._content

    @property
    def role(self) -> LLMMessageRole:
        """Retrieve the role of this instance"""
        return self._role

    def is_of_role(self, role: LLMMessageRole) -> bool:
        """Check the role of this instance"""
        return self._role == role

    @staticmethod
    def from_chat_message(chat_message: ChatMessage) -> Optional["LLMMessage"]:
        """Convert :class:`~.ChatMessage` into :class:`.LLMMessage`"""
        if chat_message.kind == ChatMessageKind.User:
            return LLMMessage.user_message(chat_message.message)
        elif chat_message.kind == ChatMessageKind.Agent:
            return LLMMessage.assistant_message(chat_message.message)
        return None

    @staticmethod
    def from_chat_messages(messages: Iterable[ChatMessage]) -> List["LLMMessage"]:
        m = map(LLMMessage.from_chat_message, messages)
        return [msg for msg in m if msg is not None]
