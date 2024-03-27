"""

A module that provides an abstract base class for implementing context providers.

This module defines an abstract base class `ContextProvider` which can be extended to
create concrete context providers. A context provider is responsible for executing
a query and returning a list of `ResponseReference` objects, which are data structures
that contain information about the response, such as title, URL, optional snippet,
and an optional date.

Classes:
    ContextProvider(ABC): An abstract base class for context provider implementations.

Typical usage example:

    class SearchEngineContextProvider(ContextProvider):
        def execute_impl(self, query: str, nb_results: int) -> list[ResponseReference]:
            # Implementation for querying a search engine
            pass

To implement a concrete context provider, subclass ContextProvider and implement the
`execute_impl` abstract method.


"""
import logging

from abc import ABC, abstractmethod

from .schemas import ResponseReference


class ContextProvider(ABC):
    """
    A base class providing a template for context providers with an abstract execute_impl method to be implemented.
    This class is responsible for initializing the context provider with a name, and providing the execute method that wraps the call to the concrete implementation of execute_impl. The execute method performs logging before and after the retrieval of results.
    
    Attributes:
        name (str):
             The name of the context provider which can be used for logging purposes.
    
    Methods:
        __init__(self, name:
             str)
            Initializes the ContextProvider instance with the given name.
        execute(self, query:
             str, nb_results: int) -> list[ResponseReference]
            Public method to initiate the retrieval of context-based results. It logs the start and
            completion of the operation and delegates the actual fetching of results to the
            execute_impl method.
    
    Args:
        query (str):
             The query string based on which the results need to be fetched.
        nb_results (int):
             The number of results to be retrieved.
    
    Returns:
        (list[ResponseReference]):
             A list of ResponseReference objects containing the results.
        execute_impl(self, query:
             str, nb_results: int) -> list[ResponseReference]
            An abstract method that should be implemented by the concrete context provider.
            This method is responsible for the actual retrieval of results based on the query.
    
    Args:
        query (str):
             The query string for retrieving results.
        nb_results (int):
             The desired number of results to fetch.
    
    Returns:
        (list[ResponseReference]):
             A list of ResponseReference objects with retrieved results.
            This class should not be instantiated directly but should be subclassed to provide specific context-based
            retrieval mechanisms.

    """
    name: str

    def __init__(self, name: str):
        """
        Initializes the instance with a name attribute.
        
        Args:
            name (str):
                 The name to assign to the instance.
            

        """
        self.name = name

    def execute(self, query: str, nb_results: int) -> list[ResponseReference]:
        """
        Executes a given query and retrieves a list of response references.
        This method logs the start and finish of the retrieval process from the context provider,
        then executes the implementation of the query execution and returns the results.
        
        Args:
            query (str):
                 The query string to be executed.
            nb_results (int):
                 The number of results to retrieve.
        
        Returns:
            (list[ResponseReference]):
                 A list of response references obtained from executing the query.
            

        """
        logging.info(f'name="{self.name}" message="start to retrieve results from context provider"')
        results = self.execute_impl(query=query, nb_results=nb_results)
        logging.info(f'name="{self.name}" message="finished to retrieve results from context provider"')
        return results

    @abstractmethod
    def execute_impl(self, query: str, nb_results: int) -> list[ResponseReference]:
        """
        Abstract method to be implemented by subclasses for executing a specific query.
        This method serves as a template for concrete implementations that perform a query and
        return a list of responses wrapped as `ResponseReference` objects. Subclasses are intended
        to provide their specific logic for how the query is executed and how results are collected.
        
        Args:
            query (str):
                 The search query string that needs to be executed.
            nb_results (int):
                 The number of results to return.
        
        Returns:
            (list[ResponseReference]):
                 A list of response references, encapsulating the query results.
        
        Raises:
            NotImplementedError:
                 If a subclass does not override this method.

        """
        pass
