import logging
from typing import List, Optional

from council.chains import Chain
from council.contexts import AgentContext, ChatHistory
from council.controllers import ControllerBase, BasicController
from council.evaluators import BasicEvaluator, EvaluatorBase
from council.runners import Budget, new_runner_executor
from council.skills import SkillBase
from .agent_result import AgentResult
from ..runners.budget import InfiniteBudget

logger = logging.getLogger(__name__)


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
                plan = self.controller.get_plan(context=context, chains=self.chains, budget=budget)
                logger.debug(f'message="agent controller returned {len(plan)} execution plan(s)"')

                if len(plan) == 0:
                    return AgentResult()
                for unit in plan:
                    chain = unit.chain
                    budget = unit.budget
                    logger.info(f'message="chain execution started" chain="{chain.name}" execution_unit="{unit.name}"')
                    chain_context = context.new_chain_context(unit.name)
                    if unit.initial_state is not None:
                        chain_context.current.append(unit.initial_state)
                    chain.execute(chain_context, budget)
                    logger.info(f'message="chain execution ended" chain="{chain.name}" execution_unit="{unit.name}"')

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
    def from_chain(chain: Chain) -> "Agent":
        """
        Helper function to create a new agent with a  :class:`.BasicController`, a
            :class:`.BasicEvaluator` and a single :class:`.SkillBase` wrapped into a :class:`.Chain`

        Parameters:
             chain(Chain): a chain
        Returns:
            Agent: a new instance
        """
        return Agent(controller=BasicController(), chains=[chain], evaluator=BasicEvaluator())

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
