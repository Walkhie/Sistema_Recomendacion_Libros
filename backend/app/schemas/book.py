from pydantic import BaseModel
from typing import List, Optional


class Book(BaseModel):
    id: str
    title: str
    author: str
    editorial: Optional[str] = None
    abstract: Optional[str] = None
    area_conocimiento: Optional[str] = None
    keywords: List[str] = []
    concepts: List[str] = []
    doi: Optional[str] = None
    institution: Optional[str] = None
    w_editorial_norm: float = 0.0
    w_citas_norm: float = 0.0