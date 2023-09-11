from ._message_list import MessageList


class ChatHistory(MessageList):
    @staticmethod
    def from_user_message(message: str) -> "ChatHistory":
        history = ChatHistory()
        history.add_user_message(message=message)
        return history
