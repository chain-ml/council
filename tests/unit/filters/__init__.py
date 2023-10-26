from council import ChatMessage
from council.contexts import ScoredChatMessage


def build_scored_message(text: str, score: float):
    return ScoredChatMessage(ChatMessage.agent(text), score)
