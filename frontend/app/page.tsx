"use client";

import { FormEvent, useEffect, useState } from "react";
import Image from "next/image";

import BookCard from "./components/BookCard";
import BookDetailModal from "./components/BookDetailModal";
import SearchBar from "./components/SearchBar";
import type { Book } from "./types/book";

const API_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_SEED_BOOK_ID = "UCO0088";
const DEFAULT_SEED_BOOK_TITLE = "Química general prácticas de laboratorio";
const DEFAULT_TOP_N = 6;

interface RecommendationItem {
  "Código del libro": string;
  Titulo_Final: string;
  Autor_Final: string;
  Area_Conocimiento: string;
  Nivel: string;
  Similitud_Texto: number;
  W_Editorial_Norm: number;
  W_Citas_Norm: number;
  Score_Final: number;
}

interface RecommendationResponse {
  seed_book_id: string;
  recommendations: RecommendationItem[];
}

interface LoadBooksParams {
  query?: string;
  title?: string;
  author?: string;
  category?: string;
  min_citations?: number;
  min_editorial_count?: number;
}

export default function HomePage() {
  const [books, setBooks] = useState<Book[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [filtersOpen, setFiltersOpen] = useState(false);

  const [searchInput, setSearchInput] = useState("");
  const [appliedQuery, setAppliedQuery] = useState("");
  const [showingDefaultRecommendations, setShowingDefaultRecommendations] =
    useState(true);

  const [titleFilter, setTitleFilter] = useState("");
  const [authorFilter, setAuthorFilter] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("");
  const [minCitations, setMinCitations] = useState(0);
  const [minEditorialCount, setMinEditorialCount] = useState(0);

  const [selectedBook, setSelectedBook] = useState<Book | null>(null);
  const [favoriteBooks, setFavoriteBooks] = useState<Record<string, boolean>>(
    {}
  );
  const [likedRecommendations, setLikedRecommendations] = useState<
    Record<string, boolean>
  >({});

  const fetchBookById = async (bookId: string): Promise<Book> => {
    const response = await fetch(`${API_BASE_URL}/books/${bookId}`);

    if (!response.ok) {
      throw new Error(`No se pudo obtener el libro ${bookId}`);
    }

    return response.json();
  };

  const loadRecommendedBooks = async () => {
    try {
      setLoading(true);
      setError("");
      setAppliedQuery("");
      setShowingDefaultRecommendations(true);

      const response = await fetch(
        `${API_BASE_URL}/books/${DEFAULT_SEED_BOOK_ID}/recommendations?top_n=${DEFAULT_TOP_N}`
      );

      if (!response.ok) {
        throw new Error("No se pudieron obtener las recomendaciones");
      }

      const data: RecommendationResponse = await response.json();

      const recommendedIds = data.recommendations.map(
        (item) => item["Código del libro"]
      );

      const recommendedBooks = await Promise.all(
        recommendedIds.map((bookId) => fetchBookById(bookId))
      );

      setBooks(recommendedBooks);
    } catch (err) {
      console.error(err);
      setError("Ocurrió un error al cargar las recomendaciones");
    } finally {
      setLoading(false);
    }
  };

  const loadBooks = async (params?: LoadBooksParams) => {
    try {
      setLoading(true);
      setError("");
      setShowingDefaultRecommendations(false);

      const searchParams = new URLSearchParams();

      if (params?.query) searchParams.set("query", params.query);
      if (params?.title) searchParams.set("title", params.title);
      if (params?.author) searchParams.set("author", params.author);
      if (params?.category) searchParams.set("category", params.category);

      if (params?.min_citations && params.min_citations > 0) {
        searchParams.set("min_citations", String(params.min_citations));
      }

      if (
        params?.min_editorial_count &&
        params.min_editorial_count > 0
      ) {
        searchParams.set(
          "min_editorial_count",
          String(params.min_editorial_count)
        );
      }

      const url = `${API_BASE_URL}/books${
        searchParams.toString() ? `?${searchParams.toString()}` : ""
      }`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error("No se pudieron obtener los libros");
      }

      const data: Book[] = await response.json();
      setBooks(data);
    } catch (err) {
      console.error(err);
      setError("Ocurrió un error al cargar los libros");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRecommendedBooks();
  }, []);

  useEffect(() => {
    const originalOverflow = document.body.style.overflow;

    if (selectedBook) {
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, [selectedBook]);

  const toggleFavorite = (bookId: string) => {
    setFavoriteBooks((prev) => ({
      ...prev,
      [bookId]: !prev[bookId],
    }));
  };

  const toggleLikedRecommendation = (bookId: string) => {
    setLikedRecommendations((prev) => ({
      ...prev,
      [bookId]: !prev[bookId],
    }));
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const trimmedQuery = searchInput.trim();
    setAppliedQuery(trimmedQuery);
    setSelectedBook(null);

    await loadBooks({
      query: trimmedQuery,
      title: titleFilter.trim(),
      author: authorFilter.trim(),
      category: categoryFilter.trim(),
      min_citations: minCitations,
      min_editorial_count: minEditorialCount,
    });
  };

  const handleClearFilters = async () => {
    setSearchInput("");
    setAppliedQuery("");
    setTitleFilter("");
    setAuthorFilter("");
    setCategoryFilter("");
    setMinCitations(0);
    setMinEditorialCount(0);
    setFiltersOpen(false);
    setSelectedBook(null);

    await loadRecommendedBooks();
  };

  return (
    <div className="page-shell">
      <SearchBar
        filtersOpen={filtersOpen}
        searchInput={searchInput}
        titleFilter={titleFilter}
        authorFilter={authorFilter}
        categoryFilter={categoryFilter}
        minCitations={minCitations}
        minEditorialCount={minEditorialCount}
        onToggleFilters={() => setFiltersOpen((prev) => !prev)}
        onSearchInputChange={setSearchInput}
        onTitleFilterChange={setTitleFilter}
        onAuthorFilterChange={setAuthorFilter}
        onCategoryFilterChange={setCategoryFilter}
        onMinCitationsChange={setMinCitations}
        onMinEditorialCountChange={setMinEditorialCount}
        onSubmit={handleSubmit}
        onClearFilters={handleClearFilters}
      />

      <main className="main-content">
        <div className="page-heading">
          <Image src="/home.png" alt="Inicio" width={34} height={34} />
          <h1>Página Principal: Recursos de tu interés</h1>
        </div>

        {showingDefaultRecommendations ? (
          <p className="search-summary">
            Mostrando <strong>{DEFAULT_TOP_N} recomendaciones</strong> basadas
            en <strong> {DEFAULT_SEED_BOOK_TITLE}</strong>.
          </p>
        ) : appliedQuery ? (
          <p className="search-summary">
            Resultados para: <strong>{appliedQuery}</strong>
          </p>
        ) : null}

        <div className="book-grid">
          {loading ? (
            <div className="status-message">Cargando libros...</div>
          ) : error ? (
            <div className="status-message error">{error}</div>
          ) : books.length > 0 ? (
            books.map((book) => (
              <BookCard
                key={book.id}
                book={book}
                isFavorite={Boolean(favoriteBooks[book.id])}
                onOpen={setSelectedBook}
                onToggleFavorite={toggleFavorite}
              />
            ))
          ) : (
            <div className="no-results">
              No se encontraron libros con los filtros actuales.
            </div>
          )}
        </div>
      </main>

      <BookDetailModal
        book={selectedBook}
        isOpen={Boolean(selectedBook)}
        isFavorite={
          selectedBook ? Boolean(favoriteBooks[selectedBook.id]) : false
        }
        isLiked={
          selectedBook
            ? Boolean(likedRecommendations[selectedBook.id])
            : false
        }
        onClose={() => setSelectedBook(null)}
        onToggleFavorite={toggleFavorite}
        onToggleLiked={toggleLikedRecommendation}
      />
    </div>
  );
}