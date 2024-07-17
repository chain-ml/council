from __future__ import annotations

import itertools
from concurrent import futures
from typing import Dict, List, Optional, Sequence

from council.chains import Chain, ChainBase
from council.contexts import AgentContext, Budget, ChainContext, InfiniteBudget, Monitorable, Monitored
from council.controllers import BasicController, ControllerBase, ExecutionUnit
from council.evaluators import BasicEvaluator, EvaluatorBase
from council.filters import BasicFilter, FilterBase
from council.runners import new_runner_executor
from council.skills import SkillBase

from .agent_result import AgentResult


class Agent(Monitorable):
    """
    Represents an agent that executes a set of chains to interact with the environment.
    """

    def __init__(
        self, controller: ControllerBase, evaluator: EvaluatorBase, filter: FilterBase, name: str = "agent"
    ) -> None:
        """
        Initializes the Agent object.

        Args:
            controller (ControllerBase): The controller responsible for generating execution plans.
            evaluator (EvaluatorBase): The evaluator responsible for evaluating the agent's performance.
            filter (FilterBase): The filter responsible to filter responses.
            name (str): name of the agent
        """
        super().__init__(base_type="agent")
        self.monitor.name = name

        self._controller: Monitored[ControllerBase] = self.new_monitor("controller", controller)
        self._chains: List[Monitored[ChainBase]] = self.new_monitors("chains", self.controller.chains)
        self._evaluator: Monitored[EvaluatorBase] = self.new_monitor("evaluator", evaluator)
        self._filter: Monitored[FilterBase] = self.new_monitor("filter", filter)

    @property
    def name(self) -> str:
        """
        the name of the agent
        """
        return self.monitor.name

    @property
    def controller(self) -> ControllerBase:
        """
        the controller of the agent
        """
        return self._controller.inner

    @property
    def evaluator(self) -> EvaluatorBase:
        """
        the evaluator of the agent
        """
        return self._evaluator.inner

    @property
    def filter(self) -> FilterBase:
        """
        the filter of the agent
        """
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
            context.logger.info('message="agent execution started"')
            while not context.budget.is_expired():
                with context.new_agent_context_for_new_iteration() as iteration_context:
                    context.logger.info(
                        f'message="agent iteration started" iteration="{iteration_context.iteration_count - 1}"'
                    )
                    plan = self.controller.execute(context=iteration_context.new_agent_context_for(self._controller))
                    context.logger.debug(f'message="agent controller returned {len(plan)} execution plan(s)"')

                    if len(plan) == 0:
                        return AgentResult()

                    self.execute_plan(iteration_context, plan)

                    result = self.evaluator.execute(iteration_context.new_agent_context_for(self._evaluator))
                    iteration_context.set_evaluation(result)

                    result = self.filter.execute(context=iteration_context.new_agent_context_for(self._filter))
                    context.logger.debug("controller selected %d responses", len(result))
                    if len(result) > 0:
                        return AgentResult(messages=result)

            return AgentResult()
        finally:
            context.logger.info('message="agent execution ended"')
            executor.shutdown(wait=False, cancel_futures=True)

    def execute_plan(self, iteration_context: AgentContext, plan: Sequence[ExecutionUnit]):
        executor = new_runner_executor("agent")
        fs = []
        try:
            for group in self._group_units(plan):
                fs = [executor.submit(self._execute_unit, iteration_context, unit) for unit in group]
                dones, _ = futures.wait(fs, iteration_context.budget.remaining_duration, futures.FIRST_EXCEPTION)
                # rethrow exception if any
                [d.result(0) for d in dones]
        finally:
            for f in fs:
                f.cancel()

    @staticmethod
    def _group_units(plan: Sequence[ExecutionUnit]) -> List[List[ExecutionUnit]]:
        groups: Dict[int, List[ExecutionUnit]] = {}
        for key, items in itertools.groupby(plan, lambda unit: unit.rank):
            group = groups.setdefault(key, [])
            group.extend(list(items))

        keys = list(groups.keys())
        keys.sort()

        result = []
        for key in keys:
            if key < 0:
                for item in groups[key]:
                    result.append([item])
            else:
                result.append(groups[key])

        return result

    @staticmethod
    def _execute_unit(iteration_context: AgentContext, unit: ExecutionUnit) -> None:
        with iteration_context.new_agent_context_for_execution_unit(unit.name) as context:
            chain = unit.chain
            context.logger.info(f'message="chain execution started" chain="{chain.name}" execution_unit="{unit.name}"')
            chain_context = ChainContext.from_agent_context(
                context, Monitored(f"chain({chain.name})", chain), unit.name, unit.budget
            )
            if unit.initial_state is not None:
                chain_context.append(unit.initial_state)
            chain.execute(chain_context)
            context.logger.info(f'message="chain execution ended" chain="{chain.name}" execution_unit="{unit.name}"')

    @staticmethod
    def from_skill(skill: SkillBase, chain_description: Optional[str] = None) -> Agent:
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
        chain: ChainBase, evaluator: EvaluatorBase = BasicEvaluator(), filter: FilterBase = BasicFilter()
    ) -> Agent:
        """
        Helper function to create a new agent with a  :class:`.BasicController`, a
            :class:`.BasicEvaluator` and a single :class:`.SkillBase` wrapped into a :class:`.Chain`

        Parameters:
            chain(ChainBase): a chain
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
