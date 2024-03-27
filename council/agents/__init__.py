"""


Module `__init__`

This module serves as an entry point for the package and exposes core classes pertinent to the operation of agents.

It imports and consolidates the primary classes that facilitate the creation, execution, and handling of agents within different contexts, including the evaluation and filtering of their outcomes.

Imports:
    AgentResult (class): A data structure designed to encapsulate the result of an Agent's execution, consisting of a list of scored chat messages.

    Agent (class): Represents an intelligent agent that can be monitored. It is responsible for executing actions within a given context and generating results based upon the evaluations and filters applied.

    AgentChain (class): Extends the base chain class to incorporate Agents as part of its execution process. It utilizes the associated Agent to determine the actions to be taken within the context of a chain interaction.


"""
from .agent_result import AgentResult
from .agent import Agent
from .agent_chain import AgentChain
