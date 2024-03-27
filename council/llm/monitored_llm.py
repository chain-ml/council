"""

Module monitored_llm

This module provides a class for monitoring and interacting with a Language
Learning Model (LLM) within a given operational context. It is designed to
overlay additional functionality, such as monitoring, to an existing LLM while
preserving its core interface and behavior.

Classes:
    MonitoredLLM
        A wrapper class that extends the functionality of the Monitored generic
class, tailored to work with objects of, or derived from LLMBase. It is used
to interface with the LLM, capture its activities, and manage interactions
based on the context in which it operates.

    Attributes:
        None

    Functions:
        __init__(self, name: str, llm: LLMBase)
            Initializes a MonitoredLLM instance with the specified name and the
LLM instance to be monitored. It calls the superclass's initializer with
these parameters.

        post_chat_request(self, context: ContextBase, messages: Sequence[LLMMessage], budget: Optional[Budget]=None, **kwargs: Any) -> LLMResult
            Submits a chat request to the underlying LLM, providing it with the
context of the interaction and the budget if specified. It constructs an
LLMContext from the provided context and wraps the chat request while
allowing for additional arguments to be passed through **kwargs.

    Exceptions:
        None

The module offers integration with broader aspects of operational contexts,
collects metrics, and ensures adherence to specified budgets while
interacting with the LLM, if such parameters are provided.


"""
from typing import Any, Optional, Sequence

from council.contexts import Budget, ContextBase, LLMContext, Monitored
from council.llm import LLMBase, LLMMessage, LLMResult


class MonitoredLLM(Monitored[LLMBase]):
    """
    A subclass for monitoring large language models (LLMs), providing an interface to track and manage interactions with an LLM.
    This class inherits from the generic `Monitored` class, taking a language model instance adhering to the `LLMBase` interface as the target to monitor. It facilitates the submission of chat requests to the underlying language model, encapsulating them in a monitored context.
    
    Attributes:
        name (str):
             A unique identifier for the monitored language model instance.
        llm (LLMBase):
             The instance of the large language model being monitored.
    
    Methods:
        post_chat_request:
            Submits a chat request to the underlying large language model, wrapped in a monitored context.
    
    Args:
        context (ContextBase):
             The chat context which includes the state and history details for the conversation.
        messages (Sequence[LLMMessage]):
             A sequence of messages to be sent to the language model.
        budget (Optional[Budget]):
             An optional budget instance to manage request costs. Default is None.
        **kwargs (Any):
             Additional keyword arguments that the underlying language model may accept.
    
    Returns:
        (LLMResult):
             The result returned by the large language model after processing the submitted messages.

    """

    def __init__(self, name: str, llm: LLMBase):
        """
        Initializes a new instance of the class with a specified name and language learning model (LLM).
        This constructor initializes the object with the provided name and language learning model,
        and it calls the base class constructor to handle any necessary initialization defined
        in the superclass.
        
        Args:
            name (str):
                 The name to be assigned to this instance.
            llm (LLMBase):
                 An instance of a language learning model derived from the LLMBase class.
        
        Raises:
            TypeError:
                 If 'llm' is not an instance of LLMBase.

        """
        super().__init__(name, llm)

    def post_chat_request(
        self, context: ContextBase, messages: Sequence[LLMMessage], budget: Optional[Budget] = None, **kwargs: Any
    ) -> LLMResult:
        """
        Fetches the default execution unit rank for the associated parallelism state.
        
        Returns:
            (Optional[int]):
                 The default rank of 1 if parallelism is enabled, otherwise None.

        """
        llm_context = LLMContext.from_context(context, self, budget)
        return self._inner.post_chat_request(llm_context, messages, **kwargs)
