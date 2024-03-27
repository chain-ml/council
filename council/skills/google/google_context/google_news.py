"""

Module for searching news using the Google News service.

This module contains a class `GoogleNewsSearchEngine` that extends from `ContextProvider` to perform
news searches using Google News. The search results are encapsulated in `ResponseReference` objects.
The class manages aspects like setting the search period, handling search queries, and processing the
results returned by the Google News API.

Classes:
    GoogleNewsSearchEngine : A context provider that utilizes the Google News API for searching news.

Functions:
    from_result(result: dict) -> Optional[ResponseReference]: Static method that converts a result
dictionary from Google News API to a `ResponseReference` object.



"""
from datetime import datetime
import logging

from typing import Optional, List, Any

from GoogleNews import GoogleNews

from .context_provider import ContextProvider
from .schemas import ResponseReference


class GoogleNewsSearchEngine(ContextProvider):
    """
    A search engine for fetching news articles from Google News.
    This class allows for searching Google News for articles matching a specific query,
    optionally within a given time period or between specified start and end dates.
    It automatically handles pagination of results up to a maximum configurable number
    of pages.
    
    Attributes:
        suffix (str):
             A string appended to each query made to Google News, presumably for
            filtering or specifying additional parameters for the search.
    
    Args:
        period (Optional[str]):
             A string representing the time period for which the
            news should be fetched. If provided, it sets the period for the GoogleNews instance.
        suffix (str):
             The suffix string to be appended to every search query.
        start (Optional[datetime]):
             The start date from which to fetch news articles.
            If the period is not set and start is provided, the search will be within
            the range from this start date to the end date.
        end (Optional[datetime]):
             The end date until which to fetch news articles.
            Defaults to the current datetime if not provided and start is set.
    
    Methods:
        execute_impl(query:
             str, nb_results: int) -> list[ResponseReference]:
            Executes the search query on Google News and returns a list of
            ResponseReference objects containing the search results up to
            the number requested or the actual number found if less.
        from_result(result:
             dict) -> Optional[ResponseReference]:
            A static method that processes a single search result dictionary
            from Google News and converts it into a ResponseReference object.
            Returns None if the result does not contain the necessary information.

    """

    suffix: str = ""

    def __init__(
        self, period: Optional[str], suffix: str, start: Optional[datetime] = None, end: Optional[datetime] = None
    ):
        """
        Initializes the object with period or specific time range for searching news on Google News.
        This constructor initializes a new instance with the option to specify a period for searching or a start and end date. If a period is provided,
        it configures the GoogleNews object for that specific period. If the period is not provided but start and end dates are,
        it configures the GoogleNews object for the time range specified by these dates. It also sets up other relevant properties
        and enables exception handling for the GoogleNews instance.
        
        Parameters:
            period (Optional[str]):
                 A string representing the time period for the news search. Can be '1d', '7d', etc.
            suffix (str):
                 A string appended to the end of something.
            start (Optional[datetime], optional):
                 A datetime object representing the starting date for the news search. Defaults to None.
            end (Optional[datetime], optional):
                 A datetime object representing the ending date for the news search. If not provided, the current datetime is used. Defaults to None.
        
        Raises:
            TypeError:
                 If the provided arguments are not of the expected datatype.
            

        """
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
        """
        Performs a search on Google News based on a given query and number of results expected.
        This method searches for news articles on Google News using the specified query term and collects a list of results up to the number indicated by 'nb_results'. It appends the 'suffix' to the search query, replaces spaces with '+' for URL encoding, and retrieves results across multiple pages until it reaches either the specified number of results or the maximum number of pages defined by '_max_page_num'.
        Each page of results is processed to create 'ResponseReference' instances which are returned as a list. If an exception occurs during the search process, an error is logged, and the process continues, though it might return an empty list if no results were found. Similarly, if no results are returned from the search, an info message is logged.
        
        Args:
            query (str):
                 The search term to be used in the Google News query.
            nb_results (int):
                 The number of news articles to retrieve.
        
        Returns:
            (list[ResponseReference]):
                 A list of 'ResponseReference' instances corresponding to the search results.

        """
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
        """
        
        Returns a ResponseReference object created from a result dictionary.
            The function expects a dictionary with certain expected keys (title, link, desc, and date) to initialize a ResponseReference object. It will only succeed if the title and url are not None or empty; the snippet and date can be None. Otherwise, the function returns None.
        
        Args:
            result (dict):
                 The search result dictionary where keys are expected to match those for initializing ResponseReference.
        
        Returns:
            (Optional[ResponseReference]):
                 An initialized ResponseReference object, or None if necessary keys are missing or empty.

        """
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
