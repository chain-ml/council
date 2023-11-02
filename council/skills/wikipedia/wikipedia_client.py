from typing import Any, Dict, List, Optional

import bs4
from mediawiki import MediaWiki, MediaWikiPage  # type: ignore


class WikipediaPageSection:
    """
    Represents partial content from a Wikipedia Page
    """

    def __init__(self, title: str, content: str, page_id: int):
        self._title = title
        self._content = content
        self._page_id = page_id

    @property
    def title(self) -> str:
        """
        Page title

        Returns:
            str:

        """
        return self._title

    @property
    def content(self) -> str:
        """
        Page section content

        Returns:
            str:
        """
        return self._content

    @property
    def page_id(self) -> int:
        """
        Page Id

        Returns:
            int:
        """
        return self._page_id

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary

        Returns:
            Dict[str, Any]:
        """
        return {"title": self._title, "content": self._content, "page_id": self._page_id}


class WikipediaClient:
    """
    Client to perform requests again Wikipedia
    """

    def __init__(self, url: str = "https://en.wikipedia.org/w/api.php"):
        """
        Initialize a new instance

        Args:
            url(str): Wikipedia url
        """

        self.wikipedia = MediaWiki(url=url)

    def search_pages_custom(self, text: str, count: int) -> List[WikipediaPageSection]:
        """
        Performs a custom search

        Args:
            text(str): the text to search for
            count(int): maximum number of result

        Returns:
            List[WikipediaPageSection]:
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
        Search for the text in a given page

        Args:
            page_id(int): the id of the page to search into
            text(str): the text to search for

        Returns:
            Optional[WikipediaPageSection]:
        """

        normalized_text = text.lower()
        page = self._get_page_by_id(page_id)
        if page is None:
            return None
        for section in [None] + page.sections:
            content = page.section(section)
            if normalized_text in content.lower():
                return WikipediaPageSection(title=section, content=content, page_id=page_id)
        return None

    def _get_page_by_id(self, page_id: int) -> Optional[MediaWikiPage]:
        return self.wikipedia.page(pageid=page_id)

    @staticmethod
    def _clean_text(text: str) -> str:
        return bs4.BeautifulSoup(text, "html.parser").text
