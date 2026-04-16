from fastapi import APIRouter, HTTPException
from app.services.recommender_service import RecommenderService

router = APIRouter(prefix="/books", tags=["recommendations"])
# Al instanciar esto aquí, la matriz TF-IDF se calcula y se queda guardada en RAM
recommender_service = RecommenderService() 

@router.get("/{book_id}/recommendations")
def get_recommendations(book_id: str, top_n: int = 10):
    recommendations = recommender_service.recommend(book_id=book_id, top_n=top_n)
    
    # Validar si el servicio devolvió un error (ej. libro no existe)
    if isinstance(recommendations, dict) and "error" in recommendations:
        raise HTTPException(status_code=404, detail=recommendations["error"])
        
    return {
        "seed_book_id": book_id,
        "recommendations": recommendations
    }