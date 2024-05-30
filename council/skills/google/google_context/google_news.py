import logging
from datetime import datetime
from typing import Any, List, Optional

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

    def __init__(
        self, period: Optional[str], suffix: str, start: Optional[datetime] = None, end: Optional[datetime] = None
    ):
        super().__init__("google name")
        self.google_news = GoogleNews()
        if period is not None:
            self.google_news.set_period(period)
        elif start is not None:
            if end is None:
                end = datetime.now()
            self.google_news.set_time_range(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

        self.suffix = suffix
        self.google_news.enableException(enable=True)
        self._max_page_num = 5

    def execute_impl(self, query: str, nb_results: int) -> list[ResponseReference]:
        self.google_news.clear()
        results: List[Any] = []
        try:
            self.google_news.search(f"{query} {self.suffix}".replace(" ", "+"))
            page_num = 1
            while len(results) < nb_results and page_num <= self._max_page_num:
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
        title: Optional[str] = result.get("title", None)
        if title is None or title == "":
            return None
        url: Optional[str] = result.get("link", None)
        if url is None or url == "":
            return None

        date = result.get("date", None)
        if date is None or date == "":
            return None

        snippet = result.get("desc", None)
        return ResponseReference(title=title, url=url, snippet=snippet, date=date)
