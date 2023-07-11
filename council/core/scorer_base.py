import abc
import logging
from typing import Any, Dict

from council.core.execution_context import AgentMessage

logger = logging.getLogger(__name__)


class ScorerException(Exception):
    """
    Exception raised by :class:`ScorerBase`
    """

    pass


class ScorerBase(abc.ABC):
    """
    Base class for implementing a Scorer
    """

    def score(self, message: AgentMessage) -> float:
        """
        Score the given message

        Parameters:
            message (AgentMessage): the message to be scored

        Returns:
            similarity score. The greater the value to higher the similarity

        Raises:
            SimilarityScorerException: an unexpected error occurs
        """
        try:
            return self._score(message)
        except Exception:
            logging.exception('message="execution failed"')
            raise ScorerException

    @abc.abstractmethod
    def _score(self, message: AgentMessage) -> float:
        """
        To be implemented with in derived classes with actual scoring logic
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the instance into a dictionary. May need to be overriden in derived classes
        """
        return {"type": self.__class__.__name__}
