"""

Module for interacting with Google News search through a conversational interface.

This module contains the `GoogleNewsSkill` class which integrates news searching
functionality into a conversational system using Google News Search Engine.
The skill retrieves news articles based on user queries within specified parameters
such as the search period, number of results, and optional date range.

Classes:
    GoogleNewsSkill: A skill class for searching Google News.

Typical usage example:
    google_news_skill = GoogleNewsSkill(suffix='technology', nb_results=3, period='7d')
    response = google_news_skill.execute(context)
    print(response)


"""
from typing import Optional
from datetime import datetime

import json

from council.contexts import ChatMessage, SkillContext
from .google_context import GoogleNewsSearchEngine
from .. import SkillBase


class GoogleNewsSkill(SkillBase):
    """
    A skill class for searching news via Google News API and responding in a chat context.
    This class provides the functionality to execute a news search using Google News based on
    user input from a chat context and return the search results formatted as a chat message.
    It is designed to be customizable with several optional parameters.
    
    Attributes:
        gn (GoogleNewsSearchEngine):
             Instance of GoogleNewsSearchEngine used for news searches.
        nb_results (int):
             The number of search results to retrieve.
    
    Args:
        suffix (str, optional):
             A suffix term to append to every search query.
        nb_results (int, optional):
             The number of results to return from the search. Defaults to 5.
        period (Optional[str], optional):
             The time period within which to search for news articles. Defaults to '90d'.
        start (Optional[datetime], optional):
             The starting date for the news search.
        end (Optional[datetime], optional):
             The ending date for the news search.
    
    Methods:
        execute(context:
             SkillContext) -> ChatMessage:
            Conducts a news search using the current context's last user message as the query,
            then constructs and returns a ChatMessage object with the search results.
    
    Args:
        context (SkillContext):
             The context of the chat which includes chat history.
    
    Returns:
        (ChatMessage):
             A ChatMessage object containing either the search results on success,
            or an error message on failure.

    """

    def __init__(
        self,
        suffix: str = "",
        nb_results=5,
        period: Optional[str] = "90d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ):
        """
        Initializes a new instance of a search engine object that searches Google News.
        This constructor sets up the search engine by initializing period, suffix, start and end times for the search,
        enabling exceptions and setting the maximum number of pages of results to fetch.
        
        Parameters:
            suffix (str):
                 A string to be appended to every search query. Defaults to an empty string.
            nb_results (int):
                 The maximum number of search results to return. Defaults to 5.
            period (Optional[str]):
                 The time period to search within. If not specified, the time range specified by start and end will be used. Defaults to '90d'.
            start (Optional[datetime]):
                 The start date of the time range for the search. If not specified, and if period is None, no start time is set.
            end (Optional[datetime]):
                 The end date of the time range for the search. If not specified but required by the lack of period and presence of start, defaults to now().
                This constructor inherits from the ContextProvider class and invokes its constructor with the argument 'gnews'.
            

        """
        super().__init__("gnews")
        self.gn = GoogleNewsSearchEngine(period=period, suffix=suffix, start=start, end=end)
        self.nb_results = nb_results

    def execute(self, context: SkillContext) -> ChatMessage:
        """
        Executes a query using the given context to fetch results from gnews.
        This method takes in a SkillContext object, which includes the chat history, extracts
        the last user message, and performs a query on gnews with that message. If there are results,
        it constructs a success message with the number of responses and their details. If there
        are no results, it returns an error message indicating the absence of a response.
        
        Args:
            context (SkillContext):
                 The context object that encapsulates the chat history and other relevant contextual information.
        
        Returns:
            (ChatMessage):
                 The response message which can be a success message including gnews responses
                or an error message in case of no responses.
        
        Raises:
            Any exception raised within the method should be documented here, separated by new lines.
        
        Note:
            If specific errors are expected to be caught, such as a network exception when making
            the gnews query, they should be documented in the 'Raises' section of this docstring.

        """
        prompt = context.chat_history.try_last_user_message.unwrap("no user message")
        resp = self.gn.execute(query=prompt.message, nb_results=self.nb_results)
        response_count = len(resp)
        if response_count > 0:
            return self.build_success_message(
                f"gnews {response_count} responses for {prompt.message}", json.dumps([r.dict() for r in resp])
            )
        return self.build_error_message("no response")
