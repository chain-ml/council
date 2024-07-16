import logging
from abc import ABC, abstractmethod

from .schemas import ResponseReference


class ContextProvider(ABC):

    def __init__(self, name: str) -> None:
        self.name: str = name

    def execute(self, query: str, nb_results: int) -> list[ResponseReference]:
        logging.info(f'name="{self.name}" message="start to retrieve results from context provider"')
        results = self.execute_impl(query=query, nb_results=nb_results)
        logging.info(f'name="{self.name}" message="finished to retrieve results from context provider"')
        return results

    @abstractmethod
    def execute_impl(self, query: str, nb_results: int) -> list[ResponseReference]:
        """
        Run query to find references used to provide context to the `ChatModel`

        @param query: input used to find references
        @param nb_results: maximum number of results needed
        @rtype: list of references
        """
        pass
