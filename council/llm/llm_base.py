import abc
from typing import Any, Optional, Sequence

from council.contexts import Consumption, LLMContext, Monitorable
from .llm_message import LLMMessage, LLMessageTokenCounterBase


class LLMResult:
    def __init__(self, choices: Sequence[str], consumptions: Optional[Sequence[Consumption]] = None):
        self._choices = list(choices)
        self._consumptions = list(consumptions) if consumptions is not None else []

    @property
    def first_choice(self) -> str:
        return self._choices[0]

    @property
    def choices(self) -> Sequence[str]:
        return self._choices

    @property
    def consumptions(self) -> Sequence[Consumption]:
        return self._consumptions


class LLMBase(Monitorable, abc.ABC):
    """
    Abstract base class representing a language model.
    """

    def __init__(self, token_counter: Optional[LLMessageTokenCounterBase] = None):
        super().__init__("llm")
        self._token_counter = token_counter

    def post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Sends a chat request to the language model.

        Parameters:
            context (LLMContext): a context to track execution metrics
            messages (Sequence[LLMMessage]): A list of LLMMessage objects representing the chat messages.
            **kwargs: Additional keyword arguments for the chat request.

        Returns:
            LLMResult: The response from the language model.

        Raises:
            LLMTokenLimitException: If messages exceed the maximum number of tokens.
            Exception: If an error occurs during the execution of the chat request.
        """

        if self._token_counter is not None:
            _ = self._token_counter.count_messages_token(messages=messages)

        context.logger.debug('message="starting execution of llm request"')
        try:
            with context:
                result = self._post_chat_request(context, messages, **kwargs)
                context.budget.add_consumptions(result.consumptions)
                return result
        except Exception as e:
            context.logger.exception(f'message="failed execution of llm request" exception="{e}" ')
            raise e
        finally:
            context.logger.debug('message="done execution of llm request"')

    @abc.abstractmethod
    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        pass
