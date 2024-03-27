"""

Module for initializing the WikipediaClient and WikipediaSearchSkill classes.

This module imports and provides access to the WikipediaClient and WikipediaPageSection classes from the
wikipedia_client module, and the WikipediaSearchSkill class from the wikipedia_search_skill module.
These classes are designed to interface with the Wikipedia API for searching and retrieving content.

Classes:
    - WikipediaClient: Provides methods to search for pages or sections within Wikipedia and process the results.
    - WikipediaPageSection: Represents a section of a Wikipedia page, encapsulating the title, content, and page ID.
    - WikipediaSearchSkill: A skill class utilizing WikipediaClient to execute searches based on incoming chat messages.


"""
from .wikipedia_client import WikipediaClient, WikipediaPageSection
from .wikipedia_search_skill import WikipediaSearchSkill
