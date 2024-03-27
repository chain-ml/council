"""

Module: agent

This module provides an architecture for an agent system involving controllers, chains, an
 evaluator, a filter, and execution units. The primary class in this module is the
 `Agent`, which integrates and manages these components to process and respond to context.

Classes:
    Agent: Represents an intelligent agent capable of performing tasks based on a
            specified context.

Functions:
    -

Exceptions:
    -

Attributes:
    -

The `Agent` class provides functionalities to execute a given context through a
 series of operations involving controller planning, evaluation, filtering results,
 and handling execution units across different chains. It also offers functionality to
 execute the agent based on a user-provided message and budget constraints.


"""
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
    A class that represents an Agent capable of executing a sequence of actions based on monitored components.
    This agent integrates various components such as a controller, chains of execution units,
    an evaluator for assessing the results, and a filter to refine the outcomes. It is a subclass
    of `Monitorable`, allowing to track and monitor its behavior and internal states.
    
    Attributes:
        _controller (Monitored[ControllerBase]):
             A monitored instance of a controller component.
        _chains (List[Monitored[ChainBase]]):
             A list of monitored chains of execution units.
        _evaluator (Monitored[EvaluatorBase]):
             A monitored instance of an evaluator component.
        _filter (Monitored[FilterBase]):
             A monitored instance of a filter component.
    
    Args:
        controller (ControllerBase):
             The controller component of the agent.
        evaluator (EvaluatorBase):
             The evaluator component used for assessing execution results.
        filter (FilterBase):
             The filter component used for refining execution results.
        name (str, optional):
             The name identifying the agent instance. Defaults to 'agent'.
    
    Returns:
        None

    """

    _controller: Monitored[ControllerBase]
    _chains: List[Monitored[ChainBase]]
    _evaluator: Monitored[EvaluatorBase]
    _filter: Monitored[FilterBase]

    def __init__(
        self, controller: ControllerBase, evaluator: EvaluatorBase, filter: FilterBase, name: str = "agent"
    ) -> None:
        """
        The initializer for an agent class that sets up monitoring for various components and assigns them to the agent instance.
        
        Args:
            controller (ControllerBase):
                 A ControllerBase instance to manage the agent's behavior.
            evaluator (EvaluatorBase):
                 An EvaluatorBase instance used for evaluating certain criteria or conditions.
            filter (FilterBase):
                 A FilterBase instance used for filtering data or decisions.
            name (str, optional):
                 A human-readable name for the agent's monitor. Defaults to 'agent'.
        
        Raises:
            TypeError:
                 If any of the input parameters are not of the expected base class type.
            

        """
        super().__init__(base_type="agent")
        self.monitor.name = name

        self._controller = self.new_monitor("controller", controller)
        self._chains = self.new_monitors("chains", self.controller.chains)
        self._evaluator = self.new_monitor("evaluator", evaluator)
        self._filter = self.new_monitor("filter", filter)

    @property
    def name(self) -> str:
        """
        Property that retrieves the name of the associated monitor object.
        This read-only property provides easy access to the name attribute of
        the monitor object linked to this instance. Attempting to set this
        property will result in an AttributeError, as it is intended to be
        immutable once the instance is created.
        
        Returns:
            (str):
                 The name of the monitor.

        """
        return self.monitor.name

    @property
    def controller(self) -> ControllerBase:
        """
        Property that returns the inner controller of the '_controller' attribute.
        This property provides access to the ControllerBase instance that is wrapped within the
        '_controller' attribute of the class. It allows for encapsulation where the inner workings
        of the '_controller' attribute are not directly exposed, but can be accessed through this property.
        
        Returns:
            (ControllerBase):
                 The inner controller object from the '_controller' attribute.
            

        """
        return self._controller.inner

    @property
    def evaluator(self) -> EvaluatorBase:
        """
        Property that gets the internal evaluator object from a wrapped instance.
        The evaluator property provides access to the `_evaluator.inner` attribute,
        which is assumed to be an instance of `EvaluatorBase` or a derived class.
        
        Returns:
            (EvaluatorBase):
                 The internal evaluator instance.
            

        """
        return self._evaluator.inner

    @property
    def filter(self) -> FilterBase:
        """
        Gets the inner `FilterBase` instance from the `filter` property.
        
        Returns:
            (FilterBase):
                 The inner `FilterBase` instance contained within `self._filter`.

        """
        return self._filter.inner

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Executes a given operation in the context of an agent.
        This method leverages a context manager to ensure that the execution is completed within
        the confines of the provided `AgentContext`. The actual execution logic is deferred to
        the `_execute` method, which is to be implemented by subclasses inheriting from this class.
        
        Args:
            context (AgentContext):
                 The context in which the agent will be executing its tasks. This
                object contains state and helper methods to manage the execution environment and resources.
        
        Returns:
            (AgentResult):
                 The result of the execution. This object encapsulates success or failure
                information, along with any data produced during the execution process.
        
        Raises:
            Any exceptions that could be thrown by the `_execute` method or by the context manager
            are to be handled by the caller of this `execute` method.
            

        """
        with context:
            return self._execute(context)

    def _execute(self, context: AgentContext) -> AgentResult:
        """
        def _execute(self, context: AgentContext) -> AgentResult:
        
        Executes the agent's action plan within the provided context.
        This method encapsulates the main execution logic for an agent. It receives the
        execution context and uses it to perform iterative actions until the assigned
        budget is expired or an appropriate response is generated. During each
        iteration, the agent's controller formulates an action plan, which is then
        executed. The result is evaluated and filtered to determine the best possible
        message to respond with.
        
        Args:
            context (AgentContext):
                 The context in which this agent should operate.
                Contains information about the current execution state such as the
                history of conversations and budget constraints.
        
        Returns:
            (AgentResult):
                 An object containing a sequence of scored messages,
                including methods to access the best score or attempt retrieving the
                best message, depending on the agent's performance and the remaining
                budget.
        
        Note:
            This function handles the setup and teardown of necessary resources
            such as executors, as well as logging of the execution process.
            Exceptions are not caught within the method; they should be handled by
            the caller. Executions will be stopped once the budget is fully used.
            

        """
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
        """
        Executes a given execution plan within an agent context using concurrent runners.
        This method handles the execution of a sequence of ExecutionUnits, often representing
        a set of tasks or actions that need to be performed by an agent. The execution occurs
        in a context that includes time budget considerations, and the tasks are grouped and
        submitted to a newly created RunnerExecutor for concurrent execution. Exceptions
        raised during execution are promptly rethrown, and remaining futures are cancelled
        upon completion or failure, ensuring that no residual tasks continue running beyond
        the intended scope of execution.
        
        Args:
            iteration_context (AgentContext):
                 The context in which the plan is being executed,
                including time budget information and other relevant state.
            plan (Sequence[ExecutionUnit]):
                 The sequence of execution units to be executed.
                These units represent the granular tasks that make up the overall plan.
        
        Raises:
            Future.exceptions:
                 Re-throws any exceptions encountered during the concurrent
                execution of tasks, if any occur.

        """
        executor = new_runner_executor("agent")
        fs = []
        try:
            for group in self._group_units(plan):
                fs = [executor.submit(self._execute_unit, iteration_context, unit) for unit in group]
                dones, not_dones = futures.wait(
                    fs, iteration_context.budget.remaining_duration, futures.FIRST_EXCEPTION
                )

                # rethrow exception if any
                [d.result(0) for d in dones]
        finally:
            [f.cancel() for f in fs]

    @staticmethod
    def _group_units(plan: Sequence[ExecutionUnit]) -> List[List[ExecutionUnit]]:
        """
        Groups ExecutionUnit objects by their rank attribute.
        This static method takes a sequence of ExecutionUnit instances and groups them by their `rank` attribute.
        Negative ranks are treated specially: each ExecutionUnit with a negative rank is placed in its own group. For
        non-negative ranks, ExecutionUnits with the same rank are grouped together. The groups are then ordered by rank in
        ascending order. The result is a list of lists, where each sublist contains ExecutionUnit instances of the same rank.
        
        Args:
            plan (Sequence[ExecutionUnit]):
                 A sequence of ExecutionUnit instances to be grouped.
        
        Returns:
            (List[List[ExecutionUnit]]):
                 A list of lists of ExecutionUnit instances, grouped and ordered by rank.
        
        Raises:
            KeyError:
                 If an ExecutionUnit instance does not contain a `rank` attribute.

        """
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
    def _execute_unit(iteration_context: AgentContext, unit: ExecutionUnit):
        """
        Executes a given ExecutionUnit within the context of a provided AgentContext.
        This method takes an iteration_context, which is an instance of AgentContext, and an
        ExecutionUnit object. It creates a new agent context that is specific to the execution unit, then
        initiates the execution of the unit's chain with the given context. If the unit has an initial
        state, it is appended to the chain context before the chain's execution. Information about the
        start and end of the execution is logged.
        
        Args:
            iteration_context (AgentContext):
                 The agent context in which the unit is being executed.
            unit (ExecutionUnit):
                 The specific unit of execution that will be run.
        
        Raises:
            Specific exceptions are not listed since the method is static and the actual implementation
            details which may raise exceptions are not provided within the given context.
        
        Returns:
            (None):
                 This method does not return anything as it's meant for execution side effects within
                the agent contexts such as state changes and logging.

        """
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
        Creates an `Agent` instance from a given `SkillBase` instance.
        This static method constructs a `BasicChain` with the provided `skill` and
        an optional description, then creates an `Agent` from the constructed `BasicChain`.
        
        Args:
            skill (SkillBase):
                 The skill to be used in creating the `BasicChain`.
            chain_description (Optional[str]):
                 An optional description for the `BasicChain`. If `None`,
                a default description of 'basic chain' is used.
        
        Returns:
            (Agent):
                 An instance of `Agent` that is constructed from the `BasicChain` containing the provided `skill`.

        """
        chain = Chain(name="BasicChain", description=chain_description or "basic chain", runners=[skill])
        return Agent.from_chain(chain)

    @staticmethod
    def from_chain(
        chain: ChainBase, evaluator: EvaluatorBase = BasicEvaluator(), filter: FilterBase = BasicFilter()
    ) -> Agent:
        """
        Creates an instance of an Agent with specified chain, evaluator, and filter components.
        
        Args:
            chain (ChainBase):
                 An instance of ChainBase that defines the sequence of executors to process.
            evaluator (EvaluatorBase, optional):
                 An instance of EvaluatorBase that is responsible for evaluating the results from the chain. Defaults to BasicEvaluator().
            filter (FilterBase, optional):
                 An instance of FilterBase that is responsible for filtering the evaluated results according to specific criteria. Defaults to BasicFilter().
        
        Returns:
            (Agent):
                 An instance of the Agent with the supplied components.

        """
        return Agent(controller=BasicController([chain]), evaluator=evaluator, filter=filter)

    def execute_from_user_message(self, message: str, budget: Optional[Budget] = None) -> AgentResult:
        """
        Executes an agent operation based on a user message, utilizing the provided budget for execution.
        This method creates an `AgentContext` from the user message and a given budget, defaulting to an `InfiniteBudget` if no budget is specified. It then proceeds to execute the agent action within the constructed context, returning the result of this execution.
        
        Args:
            message (str):
                 The user message from which the agent will derive execution context.
            budget (Optional[Budget]):
                 An optional budget for the execution, defaulting to `InfiniteBudget` if not provided.
        
        Returns:
            (AgentResult):
                 The result of the agent's execution from the context provided by the user message.

        """
        execution_budget = budget or InfiniteBudget()
        context = AgentContext.from_user_message(message, execution_budget)
        return self.execute(context)
