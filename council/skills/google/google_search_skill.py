"""

Module for Google Search Skill implementation.

This module contains the `GoogleSearchSkill` class which inherits from `SkillBase` and
provides functionality to perform Google searches using a search engine and return
the results in the form of a chat message.

Classes:
    GoogleSearchSkill: A skill that allows a chatbot to use Google Search Engine
to find and return search results based on user queries.

Attributes:
    google_search_skill (GoogleSearchSkill):
    An instance of the GoogleSearchSkill class, configured with a specific number
    of results to return and using environment variables to initialize the
    GoogleSearchEngine.



"""
import json

from council.contexts import ChatMessage, SkillContext
from .google_context import GoogleSearchEngine
from .. import SkillBase


class GoogleSearchSkill(SkillBase):
    """
    A class that encapsulates the skill to perform Google searches.
    This class inherits from `SkillBase` and is used to conduct searches on Google using a specified number of results.
    It initializes `GoogleSearchEngine` from the environment settings and takes into account the user's last chat message
    to execute the search. The search results are returned to the user in a chat message format.
    
    Attributes:
        nb_results (int):
             The number of search results to return. Defaults to 5.
    
    Methods:
        __init__(nb_results=5):
             Initializes the GoogleSearchSkill instance with a specified number of results.
        execute(context):
             Executes the search using the user's last chat message from the context and returns search results or an error message.

    """

    def __init__(self, nb_results=5):
        """
        Initializes a new instance of the class with the specified number of search results.
        
        Args:
            nb_results (int):
                 The number of search results that the search engine should return.
            

        """
        super().__init__("gsearch")
        self.gs = GoogleSearchEngine.from_env()
        self.nb_results = nb_results

    def execute(self, context: SkillContext) -> ChatMessage:
        """
        Executes the given context within a SkillContext and constructs a ChatMessage.
        This method takes in a `context` that encapsulates the chat history and seeks to leverage current chatting information
        to perform an execution via the internal mechanism `gs.execute`. It first retrieves the last user message from the chat history,
        then calls the `gs.execute` with the retrieved message as a query and with a specified number of results (`nb_results`). The results
        are then processed to form either a success or error ChatMessage based on the response count.
        
        Args:
            context (SkillContext):
                 An instance of SkillContext holding information about the current state of the chat,
                including the history.
        
        Returns:
            (ChatMessage):
                 A ChatMessage object encapsulating the result of the execution. A success message is returned if
                responses are found, which includes the name of the executor, the number of responses, and the original
                message query. In case no responses are elicited, an error message is returned.
        
        Raises:
            Exception:
                 If attempting to unwrap the last user message fails or if any other exceptions are raised within the
                execution or response building process.

        """
        prompt = context.chat_history.try_last_user_message.unwrap("no user message")
        resp = self.gs.execute(query=prompt.message, nb_results=self.nb_results)
        response_count = len(resp)
        if response_count > 0:
            return self.build_success_message(
                f"{self._name} {response_count} responses for {prompt.message}", json.dumps([r.dict() for r in resp])
            )
        return self.build_error_message("no response")
