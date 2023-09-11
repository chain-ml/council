from typing import Iterable, List, Sequence

from ._agent_iteration_context_store import AgentIterationContextStore
from ._cancellation_token import CancellationToken
from ._chat_history import ChatHistory
from ._chat_message import ScoredChatMessage
from ._execution_log import ExecutionLog
from ._message_collection import MessageCollection
from ._message_list import MessageList


class AgentContextStore:
    """
    Actual data storage used during the execution of an :class:`council.agents.Agent`
    """

    def __init__(self, chat_history: ChatHistory):
        self._cancellation_token = CancellationToken()
        self._chat_history = chat_history
        self._iterations: List[AgentIterationContextStore] = []
        self._log = ExecutionLog()

    @property
    def cancellation_token(self) -> CancellationToken:
        """
        the cancellation token
        """
        return self._cancellation_token

    @property
    def chat_history(self) -> ChatHistory:
        """
        the chat history
        """
        return self._chat_history

    @property
    def iterations(self) -> Sequence[AgentIterationContextStore]:
        """
        the storage for each execution iteration in the context
        """
        return self._iterations

    @property
    def current_iteration(self) -> AgentIterationContextStore:
        """
        the current iteration in the context
        """
        return self._iterations[-1]

    @property
    def execution_log(self) -> ExecutionLog:
        """
        the execution log
        """
        return self._log

    def new_iteration(self) -> None:
        """
        add a new iteration store in the context. It automatically becomes the current iteration.
        """
        iteration = AgentIterationContextStore()
        self._iterations.append(iteration)

    def chain_iterations(self, name: str) -> Iterable[MessageCollection]:
        """
        returns all the messages generated by a chain, across all iterations
        Args:
            name: the name of the chain
        """
        default = MessageList()
        for iteration in self._iterations:
            yield iteration.chains.get(name, default)

    @property
    def evaluation_history(self) -> Iterable[Sequence[ScoredChatMessage]]:
        """
        returns the result of all evaluations so far
        """
        for iteration in self._iterations:
            yield iteration.evaluator
