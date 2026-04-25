from fastapi import APIRouter, HTTPException

from app.schemas.book import Book
from app.services.catalog_service import CatalogService

router = APIRouter(prefix="/books", tags=["books"])
catalog_service = CatalogService()


@router.get("", response_model=list[Book])
def get_books(
    query: str = "",
    title: str = "",
    author: str = "",
    category: str = "",
    institution: str = "",
    language: str = "",
    min_citations: int | None = None,
    min_editorial_count: int | None = None,
):
    return catalog_service.get_all(
        query=query,
        title=title,
        author=author,
        category=category,
        institution=institution,
        language=language,
        min_citations=min_citations,
        min_editorial_count=min_editorial_count,
    )


@router.get("/{book_id}", response_model=Book)
def get_book(book_id: str):
    book = catalog_service.get_by_id(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Libro no encontrado")
    return book