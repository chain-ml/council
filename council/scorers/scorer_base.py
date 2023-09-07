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

    def score(self, context: ScorerContext, message: ChatMessage) -> float:
        """
        Score the given message

        Parameters:
            context (ScorerContext): the context for scoring
            message (ChatMessage): the message to be scored

        Returns:
            similarity score. The greater the value to higher the similarity

        Raises:
            ScorerException: an unexpected error occurs
        """
        try:
            return self._score(context, message)
        except Exception:
            logging.exception('message="execution failed"')
            raise ScorerException

    @abc.abstractmethod
    def _score(self, context: ScorerContext, message: ChatMessage) -> float:
        """
        To be implemented with in derived classes with actual scoring logic
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the instance into a dictionary. May need to be overriden in derived classes
        """
        return {"type": self.__class__.__name__}
