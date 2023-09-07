import abc
import logging

from typing import List, Any, Optional, Sequence
from .llm_message import LLMMessage, LLMessageTokenCounterBase
from council.contexts import LLMContext, Consumption
from council.monitors import Monitorable

logger = logging.getLogger(__name__)


class LLMResult:
    def __init__(self, choices: List[str], consumptions: Optional[List[Consumption]] = None):
        self._choices = choices
        self._consumptions = consumptions if consumptions is not None else []

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

    def post_chat_request(self, context: LLMContext, messages: List[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Sends a chat request to the language model.

        Parameters:
            context (LLMContext): a context to track execution metrics
            messages (List[LLMMessage]): A list of LLMMessage objects representing the chat messages.
            **kwargs: Additional keyword arguments for the chat request.

        Returns:
            LLMResult: The response from the language model.

        Raises:
            LLMTokenLimitException: If messages exceed the maximum number of tokens.
            Exception: If an error occurs during the execution of the chat request.
        """

        if self._token_counter is not None:
            _ = self._token_counter.count_messages_token(messages=messages)

        logger.debug('message="starting execution of llm request"')
        try:
            with context:
                result = self._post_chat_request(context, messages, **kwargs)
                context.budget.add_consumptions(result.consumptions)
                return result
        except Exception as e:
            logger.exception('message="failed execution of llm request"')
            raise e
        finally:
            logger.debug('message="done execution of llm request"')

    @abc.abstractmethod
    def _post_chat_request(self, context: LLMContext, messages: List[LLMMessage], **kwargs: Any) -> LLMResult:
        pass
