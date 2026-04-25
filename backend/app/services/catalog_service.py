import re
import unicodedata
from pathlib import Path

import pandas as pd


CSV_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "preprocessing"
    / "Libros_Limpios_Recomendador.csv"
)


def normalize_text(value: str) -> str:
    if value is None:
        return ""

    value = str(value)
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.lower()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_int(value) -> int:
    if pd.isna(value) or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def safe_float(value) -> float:
    if pd.isna(value) or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class CatalogService:
    def __init__(self):
        self.books = self._load_books()
        self.books_by_id = {book["id"]: book for book in self.books}

    def _load_books(self) -> list[dict]:
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"No se encontró el catálogo en: {CSV_PATH}")

        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

        books: list[dict] = []

        for _, row in df.iterrows():
            year = safe_str(row.get("Año de publicación"))
            title = safe_str(row.get("Titulo_Final")) or safe_str(row.get("Titulo"))
            authors = safe_str(row.get("Autor_Final")) or safe_str(row.get("Autor"))
            category = safe_str(row.get("Area_Conocimiento"))
            editorial = safe_str(row.get("Editorial"))
            institution = safe_str(
                row.get('Institución coeditora (separar cada institución con ";")')
            )

            book = {
                # compatibilidad con frontend actual
                "id": safe_str(row.get("Código del libro")),
                "title": title,
                "edition": year,  # temporalmente usamos el año para no romper el contrato actual
                "category": category,
                "authors": authors,
                "citations": safe_int(row.get("OpenAlex_Citations")),
                "editorialCount": safe_int(row.get("Titulos_Editorial_Area")),
                "editorialArea": editorial,

                # campos reales extra para detalle y evolución futura
                "year": year,
                "editorial": editorial,
                "doi": safe_str(row.get("DOI_Original")),
                "abstract": safe_str(row.get("Abstract")),
                "keywords": safe_str(row.get("Keywords")),
                "language": safe_str(row.get("Idioma")),
                "institution": institution,
                "matchMethod": safe_str(row.get("Metodo_Match")),
                "openAlexId": safe_str(row.get("OpenAlex_ID")),
                "editorialScore": round(safe_float(row.get("W_Editorial_Norm")), 4),
                "citationScore": round(safe_float(row.get("W_Citas_Norm")), 4),
            }

            if book["id"]:
                books.append(book)

        books.sort(key=lambda x: normalize_text(x["title"]))
        return books

    def get_all(
        self,
        query: str = "",
        title: str = "",
        author: str = "",
        category: str = "",
        institution: str = "",
        language: str = "",
        min_citations: int | None = None,
        min_editorial_count: int | None = None,
    ):
        results = self.books

        normalized_query = normalize_text(query)
        normalized_title = normalize_text(title)
        normalized_author = normalize_text(author)
        normalized_category = normalize_text(category)
        normalized_institution = normalize_text(institution)
        normalized_language = normalize_text(language)

        if normalized_query:
            searchable_fields = [
                "title",
                "authors",
                "category",
                "editorialArea",
                "edition",
                "editorial",
                "keywords",
                "abstract",
                "institution",
                "language",
                "doi",
            ]
            results = [
                book
                for book in results
                if any(
                    normalized_query in normalize_text(book.get(field, ""))
                    for field in searchable_fields
                )
            ]

        if normalized_title:
            results = [
                book
                for book in results
                if normalized_title in normalize_text(book["title"])
            ]

        if normalized_author:
            results = [
                book
                for book in results
                if normalized_author in normalize_text(book["authors"])
            ]

        if normalized_category:
            results = [
                book
                for book in results
                if normalized_category in normalize_text(book["category"])
                or normalized_category in normalize_text(book["keywords"])
            ]

        if normalized_institution:
            results = [
                book
                for book in results
                if normalized_institution in normalize_text(book["institution"])
            ]

        if normalized_language:
            results = [
                book
                for book in results
                if normalized_language in normalize_text(book["language"])
            ]

        if min_citations is not None:
            results = [
                book for book in results if book["citations"] >= min_citations
            ]

        if min_editorial_count is not None:
            results = [
                book
                for book in results
                if book["editorialCount"] >= min_editorial_count
            ]

        return results

    def get_by_id(self, book_id: str):
        return self.books_by_id.get(book_id)