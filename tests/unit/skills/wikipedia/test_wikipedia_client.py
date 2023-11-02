import unittest

from council.skills.wikipedia import WikipediaClient


class TestWikipediaSkill(unittest.TestCase):
    def test_search_page_custom(self):
        instance = WikipediaClient()
        result = instance.search_pages_custom("Python Programming Language", 3)

        assert len(result) == 3
        assert result[0].page_id == 23862
        assert result[0].title == "Python (programming language)"
        assert "programming language" in result[0].content

    def test_search_page_content(self):
        instance = WikipediaClient()
        result = instance.search_page_section(page_id=23862, text="whitespace")

        assert result.title == "Indentation"
        assert result.content.startswith("Python uses whitespace indentation,")

    def test_clean_text(self):
        text = (
            '<span class="searchmatch">Python</span> is a high-level, general-purpose'
            ' <span class="searchmatch">programming</span> <span class="searchmatch">language</span>.'
            " Its design philosophy emphasizes code readability with the use of significant indentation"
        )
        result = WikipediaClient._clean_text(text)
        assert "programming language" in result
