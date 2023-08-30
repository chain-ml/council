import logging
from typing import Optional

from council.chains import Chain
from council.contexts import AgentContext, ChatHistory
from council.controllers import ControllerBase, BasicController, ExecutionUnit
from council.evaluators import BasicEvaluator, EvaluatorBase
from council.runners import Budget, new_runner_executor
from council.skills import SkillBase
from .agent_result import AgentResult
from ..filters import FilterBase, BasicFilter
from ..runners.budget import InfiniteBudget

logger = logging.getLogger(__name__)


class Agent:
    """
    Represents an agent that executes a set of chains to interact with the environment.

    Attributes:
        controller (ControllerBase): The controller responsible for generating execution plans.
        evaluator (EvaluatorBase): The evaluator responsible for evaluating the agent's performance.
    """

    controller: ControllerBase
    evaluator: EvaluatorBase

    def __init__(self, controller: ControllerBase, evaluator: EvaluatorBase, filter: FilterBase) -> None:
        """
        Initializes the Agent object.

        Args:
            controller (ControllerBase): The controller responsible for generating execution plans.
            evaluator (EvaluatorBase): The evaluator responsible for evaluating the agent's performance.
            filter (FilterBase): The filter responsible to filter responses.
        """
        self.controller = controller
        self.evaluator = evaluator
        self._filter = filter

    def execute(self, context: AgentContext, budget: Optional[Budget] = None) -> AgentResult:
        """
        Executes the agent's chains based on the provided context and budget.

        Args:
            context (AgentContext): The context for executing the chains.
            budget (Optional[Budget]): The budget for agent execution. Defaults to :meth:`Budget.default` if `None`

        Returns:
            AgentResult:

        Raises:
            None
        """
        executor = new_runner_executor("agent")
        budget = budget or Budget.default()
        try:
            logger.info('message="agent execution started"')
            while not budget.is_expired():
                logger.info(f'message="agent iteration started" iteration="{len(context.evaluationHistory)+1}"')
                plan = self.controller.execute(context=context, budget=budget)
                logger.debug(f'message="agent controller returned {len(plan)} execution plan(s)"')

                if len(plan) == 0:
                    return AgentResult()
                for unit in plan:
                    budget = self._execute_unit(context, unit)

                result = self.evaluator.execute(context, budget)
                context.evaluationHistory.append(result)

                result = self._filter.execute(context=context, budget=budget)
                logger.debug("controller selected %d responses", len(result))
                if len(result) > 0:
                    return AgentResult(messages=result)

            return AgentResult()
        finally:
            logger.info('message="agent execution ended"')
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _execute_unit(context: AgentContext, unit: ExecutionUnit) -> Budget:
        chain = unit.chain
        budget = unit.budget
        logger.info(f'message="chain execution started" chain="{chain.name}" execution_unit="{unit.name}"')
        chain_context = context.new_chain_context(unit.name)
        if unit.initial_state is not None:
            chain_context.current.append(unit.initial_state)
        chain.execute(chain_context, budget)
        logger.info(f'message="chain execution ended" chain="{chain.name}" execution_unit="{unit.name}"')
        return budget

    @staticmethod
    def from_skill(skill: SkillBase, chain_description: Optional[str] = None) -> "Agent":
        """
        Helper function to create a new agent with a  :class:`.BasicController`, a
            :class:`.BasicEvaluator` and a single :class:`.SkillBase` wrapped into a :class:`.Chain`

        Parameters:
             skill(SkillBase): a skill
             chain_description(str): Optional, chain description
        Returns:
            Agent: a new instance
        """
        chain = Chain(name="BasicChain", description=chain_description or "basic chain", runners=[skill])
        return Agent.from_chain(chain)

    @staticmethod
    def from_chain(
        chain: Chain, evaluator: EvaluatorBase = BasicEvaluator(), filter: FilterBase = BasicFilter()
    ) -> "Agent":
        """
        Helper function to create a new agent with a  :class:`.BasicController`, a
            :class:`.BasicEvaluator` and a single :class:`.SkillBase` wrapped into a :class:`.Chain`

        Parameters:
            chain(Chain): a chain
            evaluator(EvaluatorBase): the Agent evaluator
            filter(FilterBase): the Agent response filter
        Returns:
            Agent: a new instance
        """
        return Agent(controller=BasicController([chain]), evaluator=evaluator, filter=filter)

    def execute_from_user_message(self, message: str, budget: Optional[Budget] = None) -> AgentResult:
        """
        Helper function that executes an agent with a simple user message.

        Parameters:
            message(str): the user message
            budget (Budget): the budget for the agent execution
        Returns:
             AgentResult:
        """
        execution_budget = budget or InfiniteBudget()
        context = AgentContext(ChatHistory.from_user_message(message))
        return self.execute(context, budget=execution_budget)
