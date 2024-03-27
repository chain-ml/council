"""

Module __init__.

This module serves as the initializer for various classes that provide a structured context for different
components such as agents, skills, scorers, and messages within a conversational system or chat environment.
It imports and enables access to classes encompassing contexts for agent execution, message handling,
budgeting, logging, and various utilitarian constructs required for monitoring and executing
conversational interactions.

The module defines the following core components required for managing conversational contexts:

- AgentContext: Maintains the context related to an agent's conversation session, including the state
  and history of interactions, as well as budgeting information.

- AgentContextStore: Holds the necessary information related to an agent's context across different
  iterations or phrases of a conversation.

- AgentIterationContextStore: Represents storage for a single iteration of a conversation, helping to
  track the changes within a specific phase.

- Budget, Consumption, and InfiniteBudget: Provide mechanisms to manage and track the resource
  constraints, such as time and computational consumption, allowing for resource-aware conversation
  execution.

- CancellationToken: Implements a way to signal cancellation of operation, enabling responsive interruption
  of conversation processes should certain conditions be met.

- ChainContext and SkillContext: Specialized context classes for handling sequences of conversational
  messages (chains) and for encapsulating the execution environment of specific skills, respectively.

- ChatHistory, ChatMessage, ChatMessageKind, and ScoredChatMessage: Concerned with the structure
  and classification of messages exchanged in conversation, along with a means to score messages.

- CompositeMessageCollection: Allows the construction of a composite view over multiple message collections.

- ContextBase, ContextLogger, ExecutionContext: Establish the base functionality for contextual execution
  and logging, providing tools to append execution insights and track conversation progress.

- ExecutionLog, ExecutionLogEntry: Comprise a logging mechanism to record and preserve the execution
  flow and performance of conversational elements.

- LLMContext, ScorerContext: Define contexts tailored for language model execution and scoring components
  within the conversation system.

- MessageCollection, MessageList: Define the means to collect and manage messages, offering iteration and
  filtering capabilities within the conversational context.

- Monitor, Monitorable, Monitored: Framework for introspection and monitoring of the components within
  the system, permitting a structured representation of monitorable elements and their properties.

- MonitoredBudget: An extension of the Budget class that enables logging of resource consumption as part
  of execution tracking.

- IterationContext: Encapsulates the notion of an iterative step within a conversation, storing its index and
  associated data, enabling contextual iteration management.



"""

from ._agent_context import AgentContext
from ._agent_context_store import AgentContextStore
from ._agent_iteration_context_store import AgentIterationContextStore
from ._budget import Budget, Consumption, InfiniteBudget
from ._cancellation_token import CancellationToken
from ._chain_context import ChainContext
from ._chat_history import ChatHistory
from ._chat_message import ChatMessage, ChatMessageKind, ScoredChatMessage
from ._composite_message_collection import CompositeMessageCollection
from ._context_base import ContextBase
from ._context_logger import ContextLogger
from ._execution_context import ExecutionContext
from ._execution_log import ExecutionLog
from ._execution_log_entry import ExecutionLogEntry
from ._llm_context import LLMContext
from ._message_collection import MessageCollection
from ._message_list import MessageList
from ._monitor import Monitor
from ._monitorable import Monitorable
from ._monitored import Monitored
from ._monitored_budget import MonitoredBudget
from ._scorer_context import ScorerContext
from ._skill_context import IterationContext, SkillContext
