from typing import Iterable, List, Sequence

from ._agent_iteration_context_store import AgentIterationContextStore
from ._cancellation_token import CancellationToken
from ._chat_history import ChatHistory
from ._chat_message import ScoredChatMessage
from ._execution_log import ExecutionLog
from ._message_collection import MessageCollection
from ._message_list import MessageList


class AgentContextStore:
    def __init__(self, chat_history: ChatHistory):
        self._cancellation_token = CancellationToken()
        self._chat_history = chat_history
        self._iterations: List[AgentIterationContextStore] = []
        self._log = ExecutionLog()

        # to be deprecated
        self._evaluation_history: List[List[ScoredChatMessage]] = []

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._cancellation_token

    @property
    def chat_history(self) -> ChatHistory:
        return self._chat_history

    @property
    def iterations(self) -> Sequence[AgentIterationContextStore]:
        return self._iterations

    @property
    def current_iteration(self) -> AgentIterationContextStore:
        return self._iterations[-1]

    @property
    def execution_log(self) -> ExecutionLog:
        return self._log

    def new_iteration(self):
        iteration = AgentIterationContextStore()
        self._iterations.append(iteration)
        self._evaluation_history.append(iteration.evaluator)

    def chain_iterations(self, name: str) -> Iterable[MessageCollection]:
        default = MessageList()
        for iteration in self._iterations:
            yield iteration.chains.get(name, default)

    @property
    def evaluation_history(self) -> Sequence[List[ScoredChatMessage]]:
        return self._evaluation_history
