import re
import unicodedata

from app.data.mock_books import MOCK_BOOKS


def normalize_text(value: str) -> str:
    if not value:
        return ""

    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.lower()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


class CatalogService:
    def __init__(self):
        self.books = MOCK_BOOKS

    def get_all(
        self,
        query: str = "",
        title: str = "",
        author: str = "",
        category: str = "",
        min_citations: int | None = None,
        min_editorial_count: int | None = None,
    ):
        results = self.books

        normalized_query = normalize_text(query)
        normalized_title = normalize_text(title)
        normalized_author = normalize_text(author)
        normalized_category = normalize_text(category)

        if normalized_query:
            results = [
                book
                for book in results
                if normalized_query in normalize_text(book["title"])
                or normalized_query in normalize_text(book["authors"])
                or normalized_query in normalize_text(book["category"])
                or normalized_query in normalize_text(book["editorialArea"])
                or normalized_query in normalize_text(book["edition"])
            ]

        if normalized_title:
            results = [
                book
                for book in results
                if normalized_title in normalize_text(book["title"])
            ]

        if normalized_author:
            results = [
                book
                for book in results
                if normalized_author in normalize_text(book["authors"])
            ]

        if normalized_category:
            results = [
                book
                for book in results
                if normalized_category in normalize_text(book["category"])
                or normalized_category in normalize_text(book["editorialArea"])
            ]

        if min_citations is not None:
            results = [
                book for book in results if book["citations"] >= min_citations
            ]

        if min_editorial_count is not None:
            results = [
                book
                for book in results
                if book["editorialCount"] >= min_editorial_count
            ]

        return results

    def get_by_id(self, book_id: str):
        for book in self.books:
            if book["id"] == book_id:
                return book
        return None