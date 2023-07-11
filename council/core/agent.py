import logging
from typing import List, Optional
from collections.abc import Sequence

from council.core.budget import Budget
from council.core.chain import Chain
from council.core.controller_base import ControllerBase
from council.core.evaluator_base import EvaluatorBase
from council.core.execution_context import AgentContext, ScoredAgentMessage, AgentMessage
from council.core.runners import new_runner_executor
from council.utils import Option

logger = logging.getLogger(__name__)


class AgentResult:
    """
    Represent the execution result of an :class:`Agent`
    """

    _messages: List[ScoredAgentMessage]

    def __init__(self, messages: Optional[List[ScoredAgentMessage]] = None):
        """
        Initialize a new instance.

        Parameters:
            messages(Optional[List[ScoredAgentMessage]]): an optional list of messages
        """
        self._messages = messages if messages is not None else []

    @property
    def messages(self) -> Sequence[ScoredAgentMessage]:
        """
        An unordered list of messages, with their scores.

        Returns:
            Sequence[ScoredAgentMessage]:
        """
        return self._messages

    @property
    def best_message(self) -> AgentMessage:
        """
        The message with the highest score. If multiple messages have the highest score, the first one is returned.

        Returns:
            AgentMessage:

        Raises:
            ValueError: there is no messages
        """
        return max(self._messages, key=lambda item: item.score).message

    @property
    def try_best_message(self) -> Option[AgentMessage]:
        """
        The message with the highest score, if any. See :meth:`best_message` for more details

        Returns:
            Option[AgentMessage]: the message with the highest score, wrapped into :meth:`.Option.some`, if some,
                :meth:`.Option.none` otherwise
        """
        if len(self._messages) == 0:
            return Option.none()
        return Option.some(self.best_message)


class Agent:
    """
    Represents an agent that executes a set of chains to interact with the environment.

    Attributes:
        controller (ControllerBase): The controller responsible for generating execution plans.
        chains (List[Chain]): The list of chains that the agent executes.
        evaluator (EvaluatorBase): The evaluator responsible for evaluating the agent's performance.
    """

    controller: ControllerBase
    chains: List[Chain]
    evaluator: EvaluatorBase

    def __init__(self, controller: ControllerBase, chains: List[Chain], evaluator: EvaluatorBase) -> None:
        """
        Initializes the Agent object.

        Args:
            controller (ControllerBase): The controller responsible for generating execution plans.
            chains (List[Chain]): The list of chains that the agent executes.
            evaluator (EvaluatorBase): The evaluator responsible for evaluating the agent's performance.
        """
        self.controller = controller
        self.chains = chains
        self.evaluator = evaluator

    def execute(self, context: AgentContext, budget: Budget) -> AgentResult:
        """
        Executes the agent's chains based on the provided context and budget.

        Args:
            context (AgentContext): The context for executing the chains.
            budget (Budget): The budget for agent execution.

        Returns:
            AgentResult:

        Raises:
            None
        """
        executor = new_runner_executor("agent")
        try:
            logger.info('message="agent execution started"')
            while not budget.is_expired():
                logger.info(f'message="agent iteration started" iteration="{len(context.evaluationHistory)+1}"')
                plan = self.controller.get_plan(context=context, chains=self.chains, budget=budget)
                logger.debug(f'message="agent controller returned {len(plan)} execution plan(s)"')

                if len(plan) == 0:
                    return AgentResult()
                for unit in plan:
                    chain = unit.chain
                    budget = unit.budget
                    logger.info(f'message="chain execution started" chain="{chain.name}"')
                    chain_context = context.new_chain_context(chain.name)
                    chain.execute(chain_context, budget)
                    logger.info(f'message="chain execution ended" chain="{chain.name}"')

                result = self.evaluator.execute(context, budget)
                context.evaluationHistory.append(result)

                result = self.controller.select_responses(context)
                logger.debug("controller selected %d responses", len(result))
                if len(result) > 0:
                    return AgentResult(messages=result)

            return AgentResult()
        finally:
            logger.info('message="agent execution ended"')
            executor.shutdown(wait=False, cancel_futures=True)
