from abc import ABC
from typing import Optional, Any

from council.utils import OptionException, read_env_str, MissingEnvVariableException

from .context_provider import ContextProvider
from .schemas import ResponseReference

from googleapiclient.discovery import build


class GoogleSearchEngine(ContextProvider, ABC):
    """
    A class that represents a Google search engine and provides functionality to execute searches and retrieve results.

    Attributes:
        suffix (str): A suffix for the search engine. Default is an empty string.

    Notes:
        GOOGLE_API_KEY environment variable needs to be set
        GOOGLE_SEARCH_ENGINE_ID environment variable needs to be set
    """

    suffix: str = ""

    def __init__(self, api_key: str, engine_id: str):
        super().__init__("gsearch")
        self._engine_id = engine_id
        self._service = build("customsearch", "v1", developerKey=api_key)

    def execute_impl(self, query: str, nb_results: int) -> list[ResponseReference]:
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
        title = result.get("title", None)
        url = result.get("link", None)
        if title is not None and url is not None:
            snippet = result.get("snippet", None)
            return ResponseReference(title=title, url=url, snippet=snippet, date=None)

        return None

    @classmethod
    def from_env(cls) -> Optional["GoogleSearchEngine"]:
        try:
            api_key: str = read_env_str("GOOGLE_API_KEY").unwrap()
            engine_id: str = read_env_str("GOOGLE_SEARCH_ENGINE_ID").unwrap()
        except (MissingEnvVariableException, OptionException):
            return None
        return GoogleSearchEngine(api_key=api_key, engine_id=engine_id)
