from app.data.mock_books import MOCK_BOOKS


class CatalogService:
    def __init__(self):
        self.books = MOCK_BOOKS

    def get_all(self, query: str = ""):
        if not query:
            return self.books

        q = query.lower()
        return [
            book for book in self.books
            if q in book["title"].lower()
            or q in book["author"].lower()
            or q in book["area_conocimiento"].lower()
            or any(q in kw.lower() for kw in book["keywords"])
        ]

    def get_by_id(self, book_id: str):
        for book in self.books:
            if book["id"] == book_id:
                return book
        return None