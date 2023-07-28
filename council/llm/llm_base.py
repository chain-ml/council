import abc
import logging

from typing import List, Any, Optional
from .llm_message import LLMMessage, LLMessageTokenCounterBase

logger = logging.getLogger(__name__)


class LLMBase(abc.ABC):
    """
    Abstract base class representing a language model.
    """

    def __init__(self, token_counter: Optional[LLMessageTokenCounterBase] = None):
        self._token_counter = token_counter

    def post_chat_request(self, messages: List[LLMMessage], **kwargs: Any) -> List[str]:
        """
        Sends a chat request to the language model.

        Parameters:
            messages (List[LLMMessage]): A list of LLMMessage objects representing the chat messages.
            **kwargs: Additional keyword arguments for the chat request.

        Returns:
            str: The response from the language model.

        Raises:
            LLMTokenLimitException: If messages exceed the maximum number of tokens.
            Exception: If an error occurs during the execution of the chat request.
        """

        if self._token_counter is not None:
            _ = self._token_counter.count_messages_token(messages=messages)

        logger.debug('message="starting execution of llm request"')
        try:
            return self._post_chat_request(messages, **kwargs)
        except Exception as e:
            logger.exception('message="failed execution of llm request"')
            raise e
        finally:
            logger.debug('message="done execution of llm request"')

    @abc.abstractmethod
    def _post_chat_request(self, messages: List[LLMMessage], **kwargs: Any) -> List[str]:
        pass
