from pydantic import BaseModel


class Book(BaseModel):
    id: str
    title: str
    edition: str
    category: str
    authors: str
    citations: int
    editorialCount: int
    editorialArea: str