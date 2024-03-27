"""

A module that provides an interface to Google's Custom Search Engine (CSE).

This module defines GoogleSearchEngine, a class that extends ContextProvider and interacts
with the Google Custom Search JSON API. Users can create instances of GoogleSearchEngine using
either direct initialization with API credentials or the factory method `from_env` that utilizes
environment variables. It uses the 'council.utils' package for reading environment variables and
dealing with potential exceptions.

The GoogleSearchEngine allows for executing search queries, returning a list of
ResponseReference objects, which contain the title, URL, optional snippet, and optional date for
each search result. The actual search is implemented in the `execute_impl` method, which
constructs the actual search query by adding an optional suffix to the user's query, then calls
the CSE API.

The `from_metadata` method is responsible for extracting relevant information from the individual
search result metadata returned by the CSE API, and converting it into a ResponseReference object.

Classes and Properties:
- GoogleSearchEngine: Extends ContextProvider, encapsulating the connection with Google CSE.
  - suffix: class property that can be set to append a custom suffix to all search queries.
  - execute_impl: overrides the abstract method from ContextProvider, implementing the
    search retrieval logic using Google's CSE API.
  - from_metadata: static method to transform JSON search result data into a ResponseReference.
  - from_env: class method that provides a way to instantiate the GoogleSearchEngine using
    environment variables.

Exceptions:
- MissingEnvVariableException: Exception thrown when a required environment variable is missing.
- OptionException: Exception thrown by utility functions when an error occurs processing an
  environment variable.


"""
from __future__ import annotations

from abc import ABC
from typing import Optional, Any

from council.utils import OptionException, read_env_str, MissingEnvVariableException

from .context_provider import ContextProvider
from .schemas import ResponseReference

from googleapiclient.discovery import build


class GoogleSearchEngine(ContextProvider, ABC):
    """
    A class that encapsulates the functionality for interacting with the Google Custom Search Engine API.
    This class provides a way to perform searches and parse the results using Google's Custom Search Engine. It inherits from `ContextProvider` and the abstract base class `ABC`. It implements the abstract method `execute_impl` which executes the actual search query. It also offers a static utility method `from_metadata` for converting individual search result metadata into a `ResponseReference` object, and a class method `from_env` to instantiate a `GoogleSearchEngine` object using environment variables.
    
    Attributes:
        suffix (str):
             An optional suffix that might be appended to every query made by this search engine.
        _engine_id (str):
             A string that identifies the specific custom search engine to use for queries.
        _service (Any):
             An instance of Google's API client service for interacting with the Custom Search Engine API.
    
    Methods:
        __init__(api_key:
             str, engine_id: str): Initializes a new GoogleSearchEngine instance with the provided API key and engine ID.
        execute_impl(query:
             str, nb_results: int) -> list[ResponseReference]: Executes the search query and returns a list of `ResponseReference` objects containing the relevant search results.
        from_metadata(result:
             Any) -> Optional[ResponseReference]: Converts a single search result into a `ResponseReference` object, or returns None if essential data is missing.
        from_env() -> Optional[GoogleSearchEngine]:
             Tries to create a `GoogleSearchEngine` instance from environment variables. Returns the instance if successful or None if required environment variables are missing or invalid.

    """

    suffix: str = ""

    def __init__(self, api_key: str, engine_id: str):
        """
        Initializes a new instance of the search engine class with the specified API key and engine ID.
        
        Args:
            api_key (str):
                 The API key to authenticate and authorize the API requests.
            engine_id (str):
                 The unique identifier for the custom search engine to use.
                The constructor internally calls the superclass's initialization method with 'gsearch' as its argument and
                sets up the custom search service client using the provided API key.

        """
        super().__init__("gsearch")
        self._engine_id = engine_id
        self._service = build("customsearch", "v1", developerKey=api_key)

    def execute_impl(self, query: str, nb_results: int) -> list[ResponseReference]:
        """
        Execute a modified query on the custom search engine and retrieve a list of response references.
        This method extends the base query with a suffix, if the suffix is not empty, before performing the search.
        It then processes the search results, converting the metadata returned for each item into a response reference,
        and compiles a list of such response references to be returned.
        
        Args:
            query (str):
                 The base query string to be searched.
            nb_results (int):
                 The number of search results to retrieve.
        
        Returns:
            (list[ResponseReference]):
                 A list of response references based on the search results.
        
        Note:
            The `execute_impl` function is an implementation detail and should not be called directly by users.
        
        Raises:
            It can raise exceptions related to network issues, or other exceptions that may
            arise from the underlying search service API client.

        """
        q = f"{query} {self.suffix}" if len(self.suffix) > 0 else query
        response = self._service.cse().list(q=q, cx=f"{self._engine_id}", num=nb_results).execute()
        metadata: list[Any] = response.get("items", [])

        references = []
        for m in metadata:
            reference = self.from_metadata(m)
            if reference is not None:
                references.append(reference)
        return references

    @staticmethod
    def from_metadata(result: Any) -> Optional[ResponseReference]:
        """
        Fetches a `ResponseReference` object from the provided metadata.
        This static method constructs a `ResponseReference` object using title and URL extracted from the given metadata. If either title or URL is missing, the method will return None.
        
        Args:
            result (Any):
                 A dictionary-like object containing metadata from which to extract the `ResponseReference` information.
        
        Returns:
            (Optional[ResponseReference]):
                 A `ResponseReference` object if both title and URL are found in the metadata; otherwise, None.

        """
        title = result.get("title", None)
        url = result.get("link", None)
        if title is not None and url is not None:
            snippet = result.get("snippet", None)
            return ResponseReference(title=title, url=url, snippet=snippet, date=None)

        return None

    @classmethod
    def from_env(cls) -> Optional[GoogleSearchEngine]:
        """
        Read environment variables and instantiate a GoogleSearchEngine object.
        This class method attempts to read the 'GOOGLE_API_KEY' and 'GOOGLE_SEARCH_ENGINE_ID' environment
        variables, expecting both to be set. If they are not set or cannot be retrieved for some reason,
        the method returns None, indicating the GoogleSearchEngine could not be initialized from
        environment variables. If the environment variables are found, it creates a new GoogleSearchEngine
        object with the provided API key and search engine ID.
        
        Returns:
            (Optional[GoogleSearchEngine]):
                 An instance of GoogleSearchEngine if environment variables are
                successfully read, or None if any required variable is missing or cannot be parsed.
        
        Raises:
            MissingEnvVariableException:
                 If a required environment variable is missing.
            OptionException:
                 If an unexpected error occurs while parsing the environment variables.
            

        """
        try:
            api_key: str = read_env_str("GOOGLE_API_KEY").unwrap()
            engine_id: str = read_env_str("GOOGLE_SEARCH_ENGINE_ID").unwrap()
        except (MissingEnvVariableException, OptionException):
            return None
        return GoogleSearchEngine(api_key=api_key, engine_id=engine_id)
