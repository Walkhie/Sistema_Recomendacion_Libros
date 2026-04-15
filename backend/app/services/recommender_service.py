from app.data.mock_books import MOCK_BOOKS


class RecommenderService:
    def recommend(self, book_id: str, top_n: int = 10):
        filtered = [book for book in MOCK_BOOKS if book["id"] != book_id]
        return filtered[:top_n]