from typing import Optional, Any, Dict


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

    def __str__(self):
        return "ResponseReference(title=" + str(self.title) + " ,url=" + self.url + ", date=" + self.date + " )"

    def dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "date": self.date,
        }
