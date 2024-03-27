"""


Module that provides functionality to perform Wikipedia searches and return the results.

This module defines the WikipediaSearchSkill class that can understand user queries through context and perform searches in Wikipedia to find relevant pages. It encapsulates the Wikipedia client functionality and interacts with the Wikipedia API.

Classes:
    WikipediaSearchSkill(SkillBase): A skill for the Council framework that provides
        the functionality to search Wikipedia based on text extracted from user
        interactions/messages.

Typical usage example:

skill = WikipediaSearchSkill()
context = SkillContext()  # Assume context is properly initialized
response = skill.execute(context)



"""
import yaml

from council import ChatMessage, SkillContext
from council.skills import SkillBase
from .wikipedia_client import WikipediaClient


class WikipediaSearchSkill(SkillBase):
    """
    A skill for searching Wikipedia and returning a formatted response.
    The WikipediaSearchSkill class inherits from SkillBase and provides functionality
    to search Wikipedia using the last user message from the skill context. It
    formats the search results in YAML and constructs a chat message to be sent back.
    
    Attributes:
        _client (WikipediaClient):
             A client instance to interact with Wikipedia.
    
    Methods:
        __init__:
             Constructs the WikipediaSearchSkill object with a default name and initializes the client.
        execute:
             Searches Wikipedia based on the last user message and returns a formatted chat message.
    
    Args:
        name (str):
             The name of the skill. Defaults to 'WikipediaSearch'.
    
    Returns:
        (ChatMessage):
             A message containing the search results formatted in YAML.

    """

    def __init__(self, name: str = "WikipediaSearch"):
        """
        Initializes the WikipediaSearch class instance with an optional name parameter.
        This method sets up a new WikipediaSearch object, creating an instance of WikipediaClient to handle interactions with Wikipedia.
        It takes an optional `name` parameter, which can be specified to give the WikipediaSearch instance a custom name.
        If not provided, the name defaults to 'WikipediaSearch'.
        
        Args:
            name (str, optional):
                 A string representing the name for the WikipediaSearch instance. Defaults to 'WikipediaSearch'.
            

        """

        super().__init__(name)
        self._client = WikipediaClient()

    def execute(self, context: SkillContext) -> ChatMessage:
        """
        Executes a search operation based on the content of the last chat message within the given context.
        The method searches custom pages related to the message content, limited to a certain number.
        It then constructs a chat message containing the search results formatted in YAML within a Markdown code block.
        
        Args:
            context (SkillContext):
                 The context containing the last message from the chat session.
        
        Returns:
            (ChatMessage):
                 A chat message object that contains the search results formatted in YAML.
        
        Raises:
            UnwrapError:
                 If there is an error in retrieving the last message from context.
            SearchPagesError:
                 If there is an error while searching for the pages.
            

        """
        last_message = context.try_last_message.unwrap("last message")
        pages = self._client.search_pages_custom(last_message.message, 5)
        response = "\n".join(["```yaml", yaml.dump([p.to_dict() for p in pages]), "```"])
        return self.build_success_message(response)
