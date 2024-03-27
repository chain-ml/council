"""

A collection of classes and functions for handling agent chains within a certain execution context.

This module provides utilities for incorporating an `Agent` into a `ChainBase` object, allowing the agent to be executed as part of a chain of execution units. It defines the `AgentChain` class which inherits from `ChainBase` and wraps around an `Agent` object, delegating the execution to the agent's implementation.

Classes:
    AgentChain(ChainBase): A specialized chain that uses an `Agent` to process the execution.

Properties:
    AgentChain.agent: Retrieve the `Agent` object that is monitored by this chain.

Methods:
    AgentChain.__init__(name, description, agent): Constructor for the `AgentChain` class, initializing a new `AgentChain` object with the specified name, description, and the agent to be monitored.
    AgentChain._execute(context, _executor): Executes the wrapped `Agent` using the given `AgentContext` derived from a `ChainContext`. If the agent execution produces a result, a `ChatMessage` is appended to the context's chat history.


"""
from typing import Any, Optional

from council.chains import ChainBase
from council.contexts import AgentContext, ChainContext, ChatMessage, Monitored
from council.runners import RunnerExecutor
from .agent import Agent


class AgentChain(ChainBase):
    """
    Class that represents a chain of agents and provides functionality to manage and execute actions.
    Inherits from ChainBase and maintains an instance of an Agent that is monitored.
    
    Attributes:
        _agent (Monitored[Agent]):
             A monitored instance of an agent which will receive execution calls.
    
    Args:
        name (str):
             The name of the chain.
        description (str):
             A description for the chain.
        agent (Agent):
             An instance of an Agent that will be executed within the chain context.
        Properties:
        agent (Agent):
             Returns the inner Agent object from the monitored instance.
    
    Methods:
        _execute(context:
             ChainContext, _executor: Optional[RunnerExecutor]=None) -> Any:
            Executes the agent's logic within the given context.
            If a message is produced during execution, it is appended to the chat history.
    
    Args:
        context (ChainContext):
             The context within which the agent operates.
        _executor (Optional[RunnerExecutor]):
             An optional executor that can be used for running the agent.
            It is not used in the current implementation.
    
    Returns:
        (Any):
             The result of the agent execution.

    """

    _agent: Monitored[Agent]

    def __init__(self, name: str, description: str, agent: Agent):
        """
        Initializes a new object with the specified name, description, and agent.
        This constructor takes in a name and description for the object along with an Agent instance.
        It calls the superclass initializer and then initializes a new monitor for the agent.
        
        Args:
            name (str):
                 The name of the object.
            description (str):
                 A brief description of the object.
            agent (Agent):
                 An instance of the Agent class which will be monitored.
            

        """
        super().__init__(name, description)
        self._agent = self.new_monitor("agent", agent)

    @property
    def agent(self) -> Agent:
        """
        Property that gets the internal Agent object.
        Allows access to the wrapped `Agent` object from the underscore-prefixed
        attribute `_agent.inner`. This property ensures that the inner `Agent`
        object can be retrieved in a controlled manner, rather than directly
        accessing the private attribute.
        
        Returns:
            (Agent):
                 The inner `Agent` object wrapped by this property.
            

        """
        return self._agent.inner

    def _execute(
        self,
        context: ChainContext,
        _executor: Optional[RunnerExecutor] = None,
    ) -> Any:
        """
        
        Returns the result of agent execution within a given context.
            This method executes the attached agent with a context converted from the chat history. If the agent produces a message, that message is appended to the current context as a chat message.
        
        Args:
            context (ChainContext):
                 The context in which to execute the agent, typically containing chat history and other relevant information.
            _executor (Optional[RunnerExecutor], optional):
                 An optional executor that may be used to run the agent. Defaults to None.
        
        Returns:
            (Any):
                 The result of the agent's execution, which could be of any type depending on the agent's implementation.

        """
        result = self.agent.execute(AgentContext.from_chat_history(context.chat_history))
        maybe_message = result.try_best_message
        if maybe_message.is_some():
            message = maybe_message.unwrap()
            context.append(ChatMessage.skill(message.message, message.data, message.source, message.is_error))
