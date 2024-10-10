from __future__ import annotations

import abc
import base64
import mimetypes
from enum import Enum
from typing import Iterable, List, Optional, Sequence

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


class LLMMessageData:
    """
    Represents the data of a message.
    """

    def __init__(self, content: str, mime_type: str) -> None:
        self._content = content
        self._mime_type = mime_type

    @property
    def content(self) -> str:
        return self._content

    @property
    def mime_type(self) -> str:
        result = self._mime_type.split(":")[-1]
        return result

    @property
    def is_image(self) -> bool:
        return self._mime_type.startswith("image/")

    @property
    def is_url(self) -> bool:
        return self._mime_type.startswith("text/url")

    def __str__(self):
        return f"content length={len(self.content)}, mime_type={self.mime_type})"

    @classmethod
    def from_file(cls, path: str) -> LLMMessageData:
        """
        Add data from file to the message.
        """
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "image/unknown"

        with open(path, "rb") as f:
            return cls(content=base64.b64encode(f.read()).decode("utf-8"), mime_type=mime_type)

    @classmethod
    def from_uri(cls, uri: str) -> LLMMessageData:
        """
        Add an uri to the message.
        """
        mime_type, _ = mimetypes.guess_type(uri)
        return cls(content=uri, mime_type=f"text/url:{mime_type}")


class LLMCacheControlData(LLMMessageData):
    """
    Data class to hold cache control information for Anthropic prompt caching.
    """

    def __init__(self, content: str) -> None:
        super().__init__(content=content, mime_type="cache")
        self.cache_control = {"type": content}

    @staticmethod
    def ephemeral() -> LLMCacheControlData:
        """Returns ephemeral cache type"""
        return LLMCacheControlData(content="ephemeral")


class LLMMessage:
    """
    Represents chat messages. Used in the payload

    Args:
        role (LLMMessageRole): the role/persona the message is coming from. Could be either user, system or assistant
        content (str): the message content
        name (str): name of the author of this message
        data (Sequence[LLMMessageData]): the data associated with this message
    """

    def __init__(
        self,
        role: LLMMessageRole,
        content: str,
        name: Optional[str] = None,
        data: Optional[Sequence[LLMMessageData]] = None,
    ) -> None:
        """Initialize a new instance of LLMMessage"""
        self._role = role
        self._content = content
        self._name = name
        self._data: List[LLMMessageData] = [] if data is None else list(data)

    @staticmethod
    def system_message(
        content: str, name: Optional[str] = None, data: Optional[Sequence[LLMMessageData]] = None
    ) -> LLMMessage:
        """
        Create a new system message instance

        Parameters:
            content (str): the message content
            name (str): name of the author of this message
            data (Sequence[LLMMessageData]): list of data associated with this message
        """
        return LLMMessage(role=LLMMessageRole.System, content=content, name=name, data=data)

    @staticmethod
    def user_message(
        content: str, name: Optional[str] = None, data: Optional[Sequence[LLMMessageData]] = None
    ) -> LLMMessage:
        """
        Create a new user message instance

        Parameters:
            content (str): the message content
            name (str): name of the author of this message
            data (Sequence[LLMMessageData]): list of data associated with this message
        """
        return LLMMessage(role=LLMMessageRole.User, content=content, name=name, data=data)

    @staticmethod
    def assistant_message(content: str, name: Optional[str] = None) -> LLMMessage:
        """
        Create a new assistant message instance

        Parameters:
            content (str): the message content
            name (str): name of the author of this message
        """
        return LLMMessage(role=LLMMessageRole.Assistant, content=content, name=name)

    @property
    def data(self) -> Sequence[LLMMessageData]:
        """
        Get the list of data associated with this message
        """
        return self._data

    def add_data(self, data: LLMMessageData) -> None:
        """
        Add data to the message.
        """
        self._data.append(data)

    def add_content(self, *, path: Optional[str] = None, url: Optional[str] = None) -> None:
        """
        Add content to the message.
        """
        data: Optional[LLMMessageData] = None
        if path is not None:
            data = LLMMessageData.from_file(path=path)
        elif url is not None:
            data = LLMMessageData.from_uri(uri=url)

        if data is not None:
            self._data.append(data)

    @property
    def content(self) -> str:
        """Retrieve the content of this instance"""
        return self._content

    @property
    def name(self) -> Optional[str]:
        """Retrieve the name authoring the content of this instance"""
        return self._name

    @property
    def role(self) -> LLMMessageRole:
        """Retrieve the role of this instance"""
        return self._role

    @property
    def has_data(self) -> bool:
        """Check if this message has data associated with it"""
        return bool(self._data)

    def is_of_role(self, role: LLMMessageRole) -> bool:
        """Check the role of this instance"""
        return self._role == role

    @staticmethod
    def from_chat_message(chat_message: ChatMessage) -> Optional[LLMMessage]:
        """Convert :class:`~.ChatMessage` into :class:`.LLMMessage`"""
        if chat_message.kind == ChatMessageKind.User:
            return LLMMessage.user_message(chat_message.message)
        elif chat_message.kind == ChatMessageKind.Agent:
            return LLMMessage.assistant_message(chat_message.message)
        return None

    @staticmethod
    def from_chat_messages(messages: Iterable[ChatMessage]) -> List[LLMMessage]:
        m = map(LLMMessage.from_chat_message, messages)
        return [msg for msg in m if msg is not None]

    def format(self, role_prefix: str = "#") -> str:
        """Format message to string, including role and LLMMessageData if any"""
        parts: List[str] = [f"{role_prefix} {self.role}\n{self.content}"]
        for data in self.data:
            parts.append(f"LLMMessageData of type {data.mime_type}:\n{data.content}")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.content


class LLMessageTokenCounterBase(abc.ABC):
    @abc.abstractmethod
    def count_messages_token(self, messages: Sequence[LLMMessage]) -> int:
        """
        Counts the total number of tokens in a list of LLM messages, including assistant tokens.

        Args:
            messages (Sequence[LLMMessage]): A list of LLMMessage objects representing the messages.

        Returns:
            int: The total number of tokens, including assistant tokens.

        Raises:
            LLMTokenLimitException: If a token limit is set (0 < limit < result) and the token count exceeds the limit.

        """
        pass
