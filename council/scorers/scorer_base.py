import abc
import logging
from typing import Any, Dict

from council.contexts import ChatMessage, ScorerContext
from .scorer_exception import ScorerException
from ..monitors import Monitorable

logger = logging.getLogger(__name__)


class ScorerBase(Monitorable, abc.ABC):
    """
    Base class for implementing a Scorer
    """

    def __init__(self):
        super().__init__("scorer")

    def score(self, message: ChatMessage, context: ScorerContext) -> float:
        """
        Score the given message

        Parameters:
            message (ChatMessage): the message to be scored
            context (ScorerContext): the context for scoring

        Returns:
            similarity score. The greater the value to higher the similarity

        Raises:
            ScorerException: an unexpected error occurs
        """
        try:
            return self._score(message, context)
        except Exception:
            logging.exception('message="execution failed"')
            raise ScorerException

    @abc.abstractmethod
    def _score(self, message: ChatMessage, context: ScorerContext) -> float:
        """
        To be implemented with in derived classes with actual scoring logic
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the instance into a dictionary. May need to be overriden in derived classes
        """
        return {"type": self.__class__.__name__}
