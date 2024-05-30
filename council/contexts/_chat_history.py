from __future__ import annotations

from ._message_list import MessageList


class ChatHistory(MessageList):
    """
    represents a history of interaction, typically between and user and an agent.
    """

    @staticmethod
    def from_user_message(message: str) -> ChatHistory:
        """
        helpers function that returns a new instance containing one user message
        """
        history = ChatHistory()
        history.add_user_message(message=message)
        return history
