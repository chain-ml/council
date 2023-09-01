from typing import Dict, List, Mapping

from ._chat_message import ChatMessage, ScoredChatMessage
from ._execution_log_entry import ExecutionLogEntry
from ._message_collection import MessageCollection
from ._message_list import MessageList
from ._monitored_message_list import MonitoredMessageList


class AgentIterationContextStore:
    _chains: Dict[str, MonitoredMessageList]
    _evaluator: List[ScoredChatMessage]

    def __init__(self):
        self._chains = {}
        self._evaluator = []

    @property
    def chains(self) -> Mapping[str, MessageCollection]:
        return self._chains

    @property
    def evaluator(self) -> List[ScoredChatMessage]:
        return self._evaluator

    def ensure_chain_exists(self, name: str):
        self._chains[name] = MonitoredMessageList(MessageList())

    def append_to_chain(self, chain: str, message: ChatMessage, log_entry: ExecutionLogEntry):
        self._chains[chain].append(message, log_entry)
