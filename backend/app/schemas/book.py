from pydantic import BaseModel


class Book(BaseModel):
    id: str
    title: str
    edition: str = ""
    category: str = ""
    authors: str = ""
    citations: int = 0
    editorialCount: int = 0
    editorialArea: str = ""

    year: str = ""
    editorial: str = ""
    doi: str = ""
    abstract: str = ""
    keywords: str = ""
    language: str = ""
    institution: str = ""
    matchMethod: str = ""
    openAlexId: str = ""
    editorialScore: float = 0.0
    citationScore: float = 0.0