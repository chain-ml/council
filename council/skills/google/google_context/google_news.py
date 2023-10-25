import logging

from typing import Optional, List, Any

from GoogleNews import GoogleNews

from .context_provider import ContextProvider
from .schemas import ResponseReference


class GoogleNewsSearchEngine(ContextProvider):
    """
    A class that represents a Google News engine and provides functionality
     to execute news searches and retrieve results.

    Attributes:
        suffix (str): A suffix for the news search engine. Default is an empty string.

    """

    suffix: str = ""

    def __init__(self, period: str, suffix: str):
        super().__init__("google name")
        self.google_news = GoogleNews(period=period)
        self.suffix = suffix
        self.google_news.enableException(enable=True)

    def execute_impl(self, query: str, nb_results: int) -> list[ResponseReference]:
        self.google_news.clear()
        results: List[Any] = []
        try:
            self.google_news.search(f"{query} {self.suffix}".replace(" ", "+"))
            page_num = 1
            while len(results) < nb_results:
                self.google_news.get_page(page_num)
                google_news_result = self.google_news.results()
                if len(google_news_result) == 0:
                    break
                results.extend(google_news_result)
                page_num += 1
        except Exception as e:
            logging.error(f"An exception occurred while searching on google news {e}")
            pass

        if len(results) == 0:
            logging.info("No Google News results were found")
            return []

        reference_results = []
        for result in results[0:nb_results]:
            response_reference = self.from_result(result)
            if response_reference is not None:
                reference_results.append(response_reference)

        return reference_results

    @staticmethod
    def from_result(result: dict) -> Optional[ResponseReference]:
        title = result.get("title", None)
        url = result.get("link", None)
        if title is not None and url is not None:
            snippet = result.get("desc", None)
            date = result.get("date", None)
            return ResponseReference(title=title, url=url, snippet=snippet, date=date)

        return None
