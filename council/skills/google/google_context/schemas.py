from typing import Any, Dict, Optional


class ResponseReference:
    """Schema for reference response"""

    title: str
    url: str
    snippet: Optional[str]
    date: Optional[str]

    def __init__(self, title: str, url: str, snippet: Optional[str], date: Optional[str]):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.date = date

    def __str__(self) -> str:
        date = self.date or "Undefined"
        return f"ResponseReference(title={self.title} ,url={self.url}, date={date})"

    def dict(self) -> Dict[str, Any]:
        result = {
            "title": self.title,
            "url": self.url,
        }
        if self.snippet:
            result["snippet"] = self.snippet
        if self.date:
            result["date"] = self.date

        return result
