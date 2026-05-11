"use client";

import { FormEvent, useEffect, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";

import BookCard from "./components/BookCard";
import BookDetailModal from "./components/BookDetailModal";
import SearchBar from "./components/SearchBar";
import type { Book } from "./types/book";

import { useAuth } from "@/context/AuthContext";
import {
  getUserFavorites,
  removeFavorite,
  saveFavorite,
  type FavoriteBook,
} from "@/lib/userStore";

const API_BASE_URL = "http://127.0.0.1:8000";
const RECOMMENDATIONS_PER_FAVORITE = 6;

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

type RecommendationGroup = {
  seedBook: Book;
  recommendations: Book[];
};

function favoriteToBook(favorite: FavoriteBook): Book {
  return {
    id: favorite.bookId,
    title: favorite.title,
    edition: favorite.edition ?? favorite.year ?? "",
    category: favorite.category ?? "",
    authors: favorite.authors ?? "",
    citations: favorite.citations ?? 0,
    editorialCount: favorite.editorialCount ?? 0,
    editorialArea: favorite.editorialArea ?? "",
    year: favorite.year ?? favorite.edition ?? "",
    editorial: favorite.editorial ?? favorite.editorialArea ?? "",
    doi: favorite.doi ?? "",
    abstract: favorite.abstract ?? "",
    keywords: favorite.keywords ?? "",
    language: favorite.language ?? "",
    institution: favorite.institution ?? "",
    matchMethod: "",
    openAlexId: "",
    editorialScore: 0,
    citationScore: 0,
  };
}

function bookToFavoritePayload(book: Book) {
  return {
    id: book.id,
    title: book.title,
    edition: book.edition,
    authors: book.authors,
    category: book.category,
    year: book.year || book.edition,
    citations: book.citations,
    editorialCount: book.editorialCount,
    editorialArea: book.editorialArea,
    editorial: book.editorial,
    doi: book.doi,
    abstract: book.abstract,
    keywords: book.keywords,
    language: book.language,
    institution: book.institution,
  };
}

export default function HomePage() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();

  const [books, setBooks] = useState<Book[]>([]);
  const [recommendationGroups, setRecommendationGroups] = useState<
    RecommendationGroup[]
  >([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [filtersOpen, setFiltersOpen] = useState(false);

  const [searchInput, setSearchInput] = useState("");
  const [appliedQuery, setAppliedQuery] = useState("");
  const [searchMode, setSearchMode] = useState(false);

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

  const loadRecommendationsForFavorite = async (
    favorite: FavoriteBook
  ): Promise<RecommendationGroup> => {
    const seedBook = favoriteToBook(favorite);

    try {
      const response = await fetch(
        `${API_BASE_URL}/books/${favorite.bookId}/recommendations?top_n=${RECOMMENDATIONS_PER_FAVORITE}`
      );

      if (!response.ok) {
        throw new Error(`No se pudieron cargar recomendaciones para ${seedBook.title}`);
      }

      const data: RecommendationResponse = await response.json();

      const recommendedIds = data.recommendations.map(
        (item) => item["Código del libro"]
      );

      const recommendations = await Promise.all(
        recommendedIds.map((bookId) => fetchBookById(bookId))
      );

      return {
        seedBook,
        recommendations,
      };
    } catch (err) {
      console.error(err);

      return {
        seedBook,
        recommendations: [],
      };
    }
  };

  const loadPersonalizedRecommendations = async () => {
    if (authLoading) return;

    try {
      setLoading(true);
      setError("");
      setBooks([]);
      setRecommendationGroups([]);
      setSearchMode(false);
      setAppliedQuery("");

      if (!user) {
        setFavoriteBooks({});
        setRecommendationGroups([]);
        return;
      }

      const favorites = await getUserFavorites(user.uid);

      const favoriteMap = favorites.reduce<Record<string, boolean>>(
        (acc, favorite) => {
          acc[favorite.bookId] = true;
          return acc;
        },
        {}
      );

      setFavoriteBooks(favoriteMap);

      if (favorites.length === 0) {
        setRecommendationGroups([]);
        return;
      }

      const groups = await Promise.all(
        favorites.map((favorite) => loadRecommendationsForFavorite(favorite))
      );

      setRecommendationGroups(
        groups.filter((group) => group.recommendations.length > 0)
      );
    } catch (err) {
      console.error(err);
      setError("Ocurrió un error al cargar tus recomendaciones.");
    } finally {
      setLoading(false);
    }
  };

  const loadBooks = async (params?: LoadBooksParams) => {
    try {
      setLoading(true);
      setError("");
      setSearchMode(true);
      setRecommendationGroups([]);

      const searchParams = new URLSearchParams();

      if (params?.query) searchParams.set("query", params.query);
      if (params?.title) searchParams.set("title", params.title);
      if (params?.author) searchParams.set("author", params.author);
      if (params?.category) searchParams.set("category", params.category);

      if (params?.min_citations && params.min_citations > 0) {
        searchParams.set("min_citations", String(params.min_citations));
      }

      if (params?.min_editorial_count && params.min_editorial_count > 0) {
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
    loadPersonalizedRecommendations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authLoading, user]);

  useEffect(() => {
    const originalOverflow = document.body.style.overflow;

    if (selectedBook) {
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, [selectedBook]);

  const toggleFavorite = async (book: Book) => {
    if (!user) {
      router.push("/login");
      return;
    }

    const wasFavorite = Boolean(favoriteBooks[book.id]);

    setFavoriteBooks((prev) => {
      const next = { ...prev };

      if (wasFavorite) {
        delete next[book.id];
      } else {
        next[book.id] = true;
      }

      return next;
    });

    try {
      if (wasFavorite) {
        await removeFavorite(user.uid, book.id);
      } else {
        await saveFavorite(user.uid, bookToFavoritePayload(book));
      }
    } catch (err) {
      console.error(err);

      setFavoriteBooks((prev) => {
        const next = { ...prev };

        if (wasFavorite) {
          next[book.id] = true;
        } else {
          delete next[book.id];
        }

        return next;
      });
    }
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

    setFiltersOpen(false);
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

    await loadPersonalizedRecommendations();
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

        {searchMode && appliedQuery ? (
          <p className="search-summary">
            Resultados para: <strong>{appliedQuery}</strong>
          </p>
        ) : !searchMode && user ? (
          <p className="search-summary">
            Recomendaciones generadas a partir de tus libros favoritos.
          </p>
        ) : null}

        {loading ? (
          <div className="status-message">Cargando libros...</div>
        ) : error ? (
          <div className="status-message error">{error}</div>
        ) : searchMode ? (
          <div className="book-grid">
            {books.length > 0 ? (
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
        ) : recommendationGroups.length > 0 ? (
          <>
            <div className="home-recommendations-slider">
              {recommendationGroups.map((group) => (
                <section
                  key={group.seedBook.id}
                  className="home-recommendation-page"
                >
                  <div className="home-recommendation-header">
                    <span>Basado en tu favorito</span>
                    <h2>{group.seedBook.title}</h2>
                  </div>

                  <div className="home-recommendation-grid">
                    {group.recommendations.map((book) => (
                      <BookCard
                        key={`${group.seedBook.id}-${book.id}`}
                        book={book}
                        isFavorite={Boolean(favoriteBooks[book.id])}
                        onOpen={setSelectedBook}
                        onToggleFavorite={toggleFavorite}
                      />
                    ))}
                  </div>
                </section>
              ))}
            </div>

            {recommendationGroups.length > 1 ? (
              <p className="home-scroll-hint">
                Desliza horizontalmente para ver recomendaciones basadas en tus
                otros favoritos.
              </p>
            ) : null}
          </>
        ) : user ? (
          <div className="no-results">
            Aún no tienes favoritos para generar recomendaciones. Selecciona
            libros desde el onboarding o marca libros con el corazón.
          </div>
        ) : (
          <div className="no-results">
            Inicia sesión para ver recomendaciones personalizadas basadas en tus
            favoritos.
          </div>
        )}
      </main>

      <BookDetailModal
        book={selectedBook}
        isOpen={Boolean(selectedBook)}
        isFavorite={
          selectedBook ? Boolean(favoriteBooks[selectedBook.id]) : false
        }
        isLiked={
          selectedBook ? Boolean(likedRecommendations[selectedBook.id]) : false
        }
        onClose={() => setSelectedBook(null)}
        onToggleFavorite={toggleFavorite}
        onToggleLiked={toggleLikedRecommendation}
      />
    </div>
  );
}