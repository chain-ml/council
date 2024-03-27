"""

A module initializer for the Google News and Google Search engine classes.

This module imports the `GoogleNewsSearchEngine` class from the `google_news` submodule and the
`GoogleSearchEngine` class from the `google_search` submodule, allowing them to be utilized elsewhere in the application.

The classes are designed to provide search functionalities via Google News and the Custom Google Search API.
The `GoogleNewsSearchEngine` class is specialized for fetching news articles from Google News based on specific search queries,
while the `GoogleSearchEngine` class makes use of a custom search engine to retrieve search results for any provided query.


"""
from .google_news import GoogleNewsSearchEngine
from .google_search import GoogleSearchEngine
