from fastapi import APIRouter, HTTPException
from app.services.catalog_service import CatalogService

router = APIRouter(prefix="/books", tags=["books"])
catalog_service = CatalogService()


@router.get("")
def get_books(query: str = ""):
    return catalog_service.get_all(query=query)


@router.get("/{book_id}")
def get_book(book_id: str):
    book = catalog_service.get_by_id(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Libro no encontrado")
    return book