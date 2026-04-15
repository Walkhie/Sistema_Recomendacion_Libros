from fastapi import APIRouter
from app.services.recommender_service import RecommenderService

router = APIRouter(prefix="/books", tags=["recommendations"])
recommender_service = RecommenderService()


@router.get("/{book_id}/recommendations")
def get_recommendations(book_id: str, top_n: int = 10):
    recommendations = recommender_service.recommend(book_id=book_id, top_n=top_n)
    return {
        "seed_book_id": book_id,
        "recommendations": recommendations
    }