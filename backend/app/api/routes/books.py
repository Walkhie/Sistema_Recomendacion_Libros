from fastapi import APIRouter, HTTPException, Query

from app.schemas.book import Book
from app.services.catalog_service import CatalogService

router = APIRouter(prefix="/books", tags=["books"])

catalog_service = CatalogService()


@router.get("/topics")
def get_topics(limit: int = Query(default=40, ge=1, le=100)):
    return catalog_service.get_topics(limit=limit)


@router.get("/top-by-topics", response_model=list[Book])
def get_top_books_by_topics(
    topics: str = Query(..., description="Temas separados por coma"),
    top_n: int = Query(default=72, ge=1, le=99),
):
    topic_list = [
        topic.strip()
        for topic in topics.split(",")
        if topic.strip()
    ]

    return catalog_service.get_top_by_topics(topic_list, top_n=top_n)


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
def get_book_by_id(book_id: str):
    book = catalog_service.get_by_id(book_id)

    if not book:
        raise HTTPException(status_code=404, detail="Libro no encontrado")

    return book