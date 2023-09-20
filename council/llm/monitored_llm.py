from typing import Any, List, Optional

from council.contexts import Budget, ContextBase, LLMContext, Monitored
from council.llm import LLMBase, LLMMessage, LLMResult


class MonitoredLLM(Monitored[LLMBase]):
    """
    A convenience class that wraps and LLM into a Monitor
    """

    def __init__(self, name: str, llm: LLMBase):
        super().__init__(name, llm)

    def post_chat_request(
        self, context: ContextBase, messages: List[LLMMessage], budget: Optional[Budget] = None, **kwargs: Any
    ) -> LLMResult:
        """
        make a call to the wrapped llm, managing the creation of the context.
        See :meth:`LLMBase.post_chat_request`

        Args:
            context (ContextBase): the context of the caller
            messages (List[LLMMessage]): see :meth:`LLMBase.post_chat_request`
            budget (Optional[Budget]): an optional budget. If none, the budget from the given context is used
            **kwargs: see :meth:`LLMBase.post_chat_request`

        Returns:
            LLMResult: see :meth:`LLMBase.post_chat_request`
        """
        llm_context = LLMContext.from_context(context, self, budget)
        return self._inner.post_chat_request(llm_context, messages, **kwargs)
