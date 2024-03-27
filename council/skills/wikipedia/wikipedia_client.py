"""

Module for interacting with the Wikipedia API through a simple client interface.

This module contains the WikipediaClient class which allows users to perform custom searches and extract sections of Wikipedia pages. It also provides a utility class WikipediaPageSection to encapsulate information about individual page sections.

Classes:
    WikipediaPageSection: Encapsulation of a Wikipedia page section's title, content, and page ID.
    WikipediaClient: Client interface for searching and retrieving data from Wikipedia.

Functions:
    _get_page_by_id: Retrieve a MediaWikiPage object by its page ID.
    _clean_text: Strip HTML elements from a string to provide plain text.



"""
from typing import Any, Dict, List, Optional

import bs4
from mediawiki import MediaWiki, MediaWikiPage  # type: ignore


class WikipediaPageSection:
    """
    A representation of a section of a Wikipedia page.
    This class encapsulates details of a specific section on a Wikipedia page including the section's title, its content, and the page's identifying ID. The section's attributes are encapsulated and made accessible via property methods.
    
    Attributes:
        _title (str):
             The title of the section.
        _content (str):
             The content of the section, which may include text and markup.
        _page_id (int):
             The unique identifier for the Wikipedia page that this section belongs to.
    
    Methods:
        title:
             Property that returns the title of the section.
        content:
             Property that returns the content of the section.
        page_id:
             Property that returns the page identifier of the section.
        to_dict:
             Returns a dictionary representation of the Wikipedia page section with keys 'title', 'content', and 'page_id'.

    """

    def __init__(self, title: str, content: str, page_id: int):
        """
        Initializes a new instance of a class with a title, content, and page ID.
        
        Args:
            title (str):
                 The title for the instance. This should be a string representing the
                title of the content.
            content (str):
                 The actual content of the instance. This is a string representing the
                material or data that the instance holds.
            page_id (int):
                 An integer representing the unique identifier for the content
                instance. It is used to uniquely identify this instance within a collection or database.
        
        Attributes:
            _title (str):
                 A private attribute that holds the title of the content.
            _content (str):
                 A private attribute that contains the content.
            _page_id (int):
                 A private attribute representing the unique identifier for the instance.

        """
        self._title = title
        self._content = content
        self._page_id = page_id

    @property
    def title(self) -> str:
        """
        Gets the title of the object.
        This property method returns the title of the object. The title is a string attribute, which should hold the name or designation for the object it represents. This method is decorated as a property, meaning that it can be accessed as an attribute without the need to call it like a method.
        
        Returns:
            (str):
                 The title of the object.

        """
        return self._title

    @property
    def content(self) -> str:
        """
        Gets the content attribute of the class instance.
        This method is a property getter that returns the private _content attribute, holding data within the class instance. It is accessed through the 'content' property, allowing for a controlled way of retrieving the content value without direct access to the private variable.
        
        Returns:
            (str):
                 The content held within the _content attribute of the class instance.

        """
        return self._content

    @property
    def page_id(self) -> int:
        """
        
        Returns the page identifier (ID) of the current instance.
        
        Returns:
            (int):
                 The unique identifier (ID) of the page.

        """
        return self._page_id

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation.
        This method serializes the object's attributes into a dictionary format, which is useful for
        JSON serialization or similar operations. It includes the title, content, and page ID of the object.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary with keys 'title', 'content', and 'page_id' mapping to corresponding object attributes.

        """
        return {"title": self._title, "content": self._content, "page_id": self._page_id}


class WikipediaClient:
    """
    A client for interacting with a MediaWiki API, particularly designed for Wikipedia.
    This client is used for searching Wikipedia pages and fetching sections of those pages.
    It provides functionalities to search for pages with specific text, get a custom list
    of search results, and retrieve specific sections of a page by page ID and section text.
    
    Attributes:
        wikipedia (MediaWiki):
             An instance of the MediaWiki class initialized with the API URL
            for Wikipedia.
        Initializer:
    
    Args:
        url (str):
             The URL of the Wikipedia's MediaWiki API endpoint. Defaults to the
            English Wikipedia's API endpoint.
    
    Methods:
        search_pages_custom:
            Searches for Wikipedia pages that match the given text and returns a limited list
            of custom page sections containing the search text.
    
    Args:
        text (str):
             The text to search for within Wikipedia pages.
        count (int):
             The maximum number of page sections to return.
    
    Returns:
        (List[WikipediaPageSection]):
             A list of WikipediaPageSection objects containing
            the title, cleaned content snippets, and page ID of the search results.
        (search_page_section):
            Searches for and returns a specific section of a Wikipedia page that contains
            the given text.
    
    Args:
        page_id (int):
             The page ID of the Wikipedia page to search within.
        text (str):
             The text to search for within the specified page section.
    
    Returns:
        (Optional[WikipediaPageSection]):
             A WikipediaPageSection object containing the
            section title, full content, and page ID if a matching section is found,
            otherwise None.
        (_get_page_by_id):
            Retrieves a MediaWikiPage object for a given Wikipedia page ID.
    
    Args:
        page_id (int):
             The ID of the Wikipedia page to retrieve.
    
    Returns:
        (Optional[MediaWikiPage]):
             The MediaWikiPage object if found, otherwise None.
        (_clean_text):
            Cleans and sanitizes HTML content to plain text.
    
    Args:
        text (str):
             The HTML content to clean.
    
    Returns:
        (str):
             The sanitized plain text content.

    """

    def __init__(self, url: str = "https://en.wikipedia.org/w/api.php"):
        """
        Initializes the object with a MediaWiki instance pointing to the specified API URL.
        
        Args:
            url (str, optional):
                 The URL of the MediaWiki API. Defaults to 'https://en.wikipedia.org/w/api.php'.
        
        Attributes:
            wikipedia (MediaWiki):
                 An instance of the MediaWiki class initialized with the given API URL.

        """

        self.wikipedia = MediaWiki(url=url)

    def search_pages_custom(self, text: str, count: int) -> List[WikipediaPageSection]:
        """
        Fetches a list of Wikipedia page sections based on a search text, limiting the number of results to a specified count.
        This function performs a search request to Wikipedia using the provided text and collects
        the resulting page sections up to the specified count. Each page section includes the
        title, a cleaned version of the snippet, and the page ID.
        
        Args:
            text (str):
                 The text to search for within Wikipedia articles.
            count (int):
                 The maximum number of page sections to return.
        
        Returns:
            (List[WikipediaPageSection]):
                 A list of WikipediaPageSection objects, each containing
                a title, a cleaned snippet as content, and a page ID corresponding to the search results.
        
        Note:
            The actual number of results returned may be less than the specified count if
            fewer search results are found.
            

        """

        result = []
        search_result = self.wikipedia.wiki_request(
            {
                "action": "query",
                "list": "search",
                "srsearch": text,
            }
        )

        for item in search_result["query"]["search"]:
            title = item["title"]
            content = self._clean_text(item["snippet"])
            page_id = item["pageid"]
            result.append(WikipediaPageSection(title=title, content=content, page_id=page_id))
            if len(result) >= count:
                break
        return result

    def search_page_section(self, page_id: int, text: str) -> Optional[WikipediaPageSection]:
        """
        Search for a text within the sections of a Wikipedia page with the given page_id.
        This method looks for the given text within all sections of a specific Wikipedia page,
        identified by the page_id. It then returns a WikipediaPageSection object representing the
        section where the text was found. If the text is not found in any section, or the page does
        not exist, the method returns None.
        
        Args:
            page_id (int):
                 The ID of the Wikipedia page to be searched.
            text (str):
                 The text to search for within the page sections.
        
        Returns:
            (Optional[WikipediaPageSection]):
                 An object representing the section which contains the
                specified text, or None if the text is not found or the page does not exist.
            

        """

        normalized_text = text.lower()
        page = self._get_page_by_id(page_id)
        if page is None:
            return None
        for section in [None] + page.sections:
            content = page.section(section)
            if content is None:
                continue
            if normalized_text in content.lower():
                return WikipediaPageSection(title=section or "", content=content, page_id=page_id)
        return None

    def _get_page_by_id(self, page_id: int) -> Optional[MediaWikiPage]:
        """
        Retrieves a MediaWikiPage object corresponding to the given page ID from Wikipedia.
        
        Args:
            page_id (int):
                 The page ID for the Wikipedia page to retrieve.
        
        Returns:
            (Optional[MediaWikiPage]):
                 A MediaWikiPage object if the page exists; otherwise, None.
        
        Raises:
            MediaWikiException:
                 If any issue occurs while retrieving the page (handled internally by the wikipedia API).

        """
        return self.wikipedia.page(pageid=page_id)

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Cleans the HTML content from the provided text string and returns plain text.
        This method uses BeautifulSoup to parse the provided HTML content and extract text without any HTML tags.
        
        Args:
            text (str):
                 The text string containing HTML content to be cleaned.
        
        Returns:
            (str):
                 The cleaned text as plain text, with HTML tags removed.
            

        """
        return bs4.BeautifulSoup(text, "html.parser").text
