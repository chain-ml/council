import logging
from typing import List, Optional

from council.chains import Chain
from council.contexts import AgentContext, Budget, ChainContext, InfiniteBudget, Monitorable, Monitored
from council.controllers import BasicController, ControllerBase, ExecutionUnit
from council.evaluators import BasicEvaluator, EvaluatorBase
from council.filters import BasicFilter, FilterBase
from council.runners import new_runner_executor
from council.skills import SkillBase
from .agent_result import AgentResult

logger = logging.getLogger(__name__)


class Agent(Monitorable):
    """
    Represents an agent that executes a set of chains to interact with the environment.

    Attributes:
        controller (ControllerBase): The controller responsible for generating execution plans.
        evaluator (EvaluatorBase): The evaluator responsible for evaluating the agent's performance.
    """

    _controller: Monitored[ControllerBase]
    _chains: List[Monitored[Chain]]
    _evaluator: Monitored[EvaluatorBase]
    _filter: Monitored[FilterBase]

    def __init__(self, controller: ControllerBase, evaluator: EvaluatorBase, filter: FilterBase) -> None:
        """
        Initializes the Agent object.

        Args:
            controller (ControllerBase): The controller responsible for generating execution plans.
            evaluator (EvaluatorBase): The evaluator responsible for evaluating the agent's performance.
            filter (FilterBase): The filter responsible to filter responses.
        """
        super().__init__("agent")

        self._controller = self.new_monitor("controller", controller)
        self._chains = self.new_monitors("chains", self.controller.chains)
        self._evaluator = self.new_monitor("evaluator", evaluator)
        self._filter = self.new_monitor("filter", filter)

    @property
    def controller(self) -> ControllerBase:
        return self._controller.inner

    @property
    def evaluator(self) -> EvaluatorBase:
        return self._evaluator.inner

    @property
    def filter(self) -> FilterBase:
        return self._filter.inner

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Executes the agent's chains based on the provided context and budget.

        Args:
            context (AgentContext): The context for executing the chains.

        Returns:
            AgentResult:

        Raises:
            None
        """
        with context:
            return self._execute(context)

    def _execute(self, context: AgentContext) -> AgentResult:
        executor = new_runner_executor("agent")
        try:
            logger.info('message="agent execution started"')
            while not context.budget.is_expired():
                with context.new_agent_context_for_new_iteration() as iteration_context:
                    logger.info(
                        f'message="agent iteration started" iteration="{iteration_context.iteration_count - 1}"'
                    )
                    plan = self.controller.execute(context=iteration_context.new_agent_context_for(self._controller))
                    logger.debug(f'message="agent controller returned {len(plan)} execution plan(s)"')

                    if len(plan) == 0:
                        return AgentResult()

                    for unit in plan:
                        with iteration_context.new_agent_context_for_execution_unit(unit.name) as unit_context:
                            self._execute_unit(unit_context, unit)

                    result = self.evaluator.execute(iteration_context.new_agent_context_for(self._evaluator))
                    iteration_context.set_evaluation(result)

                    result = self.filter.execute(context=iteration_context.new_agent_context_for(self._filter))
                    logger.debug("controller selected %d responses", len(result))
                    if len(result) > 0:
                        return AgentResult(messages=result)

            return AgentResult()
        finally:
            logger.info('message="agent execution ended"')
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _execute_unit(context: AgentContext, unit: ExecutionUnit):
        chain = unit.chain
        logger.info(f'message="chain execution started" chain="{chain.name}" execution_unit="{unit.name}"')
        chain_context = ChainContext.from_agent_context(
            context, Monitored(f"chain({chain.name})", chain), unit.name, unit.budget
        )
        if unit.initial_state is not None:
            chain_context.append(unit.initial_state)
        chain.execute(chain_context)
        logger.info(f'message="chain execution ended" chain="{chain.name}" execution_unit="{unit.name}"')

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
        context = AgentContext.from_user_message(message, execution_budget)
        return self.execute(context)
