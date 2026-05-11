"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";

import BookCard from "../components/BookCard";
import BookDetailModal from "../components/BookDetailModal";
import SearchBar from "../components/SearchBar";
import type { Book } from "../types/book";

import { useAuth } from "@/context/AuthContext";
import {
  getUserFavorites,
  removeFavorite,
  saveFavorite,
  type FavoriteBook,
} from "@/lib/userStore";

const API_BASE_URL = "http://127.0.0.1:8000";

type FavoriteFilters = {
  query: string;
  title: string;
  author: string;
  category: string;
  minCitations: number;
  minEditorialCount: number;
};

function normalizeText(value?: string | number) {
  return String(value ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
}

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

export default function FavoritesPage() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();

  const [books, setBooks] = useState<Book[]>([]);
  const [favoriteBooks, setFavoriteBooks] = useState<Record<string, boolean>>(
    {}
  );
  const [likedRecommendations, setLikedRecommendations] = useState<
    Record<string, boolean>
  >({});

  const [selectedBook, setSelectedBook] = useState<Book | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [filtersOpen, setFiltersOpen] = useState(false);

  const [searchInput, setSearchInput] = useState("");
  const [titleFilter, setTitleFilter] = useState("");
  const [authorFilter, setAuthorFilter] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("");
  const [minCitations, setMinCitations] = useState(0);
  const [minEditorialCount, setMinEditorialCount] = useState(0);

  const [activeFilters, setActiveFilters] = useState<FavoriteFilters>({
    query: "",
    title: "",
    author: "",
    category: "",
    minCitations: 0,
    minEditorialCount: 0,
  });

  const fetchBookById = async (bookId: string): Promise<Book> => {
    const response = await fetch(`${API_BASE_URL}/books/${bookId}`);

    if (!response.ok) {
      throw new Error(`No se pudo obtener el libro ${bookId}`);
    }

    return response.json();
  };

  useEffect(() => {
    if (!authLoading && !user) {
      router.push("/login");
    }
  }, [authLoading, user, router]);

  useEffect(() => {
    let active = true;

    async function loadFavorites() {
      if (authLoading) return;

      if (!user) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError("");

        const favorites = await getUserFavorites(user.uid);

        const favoriteMap = favorites.reduce<Record<string, boolean>>(
          (acc, favorite) => {
            acc[favorite.bookId] = true;
            return acc;
          },
          {}
        );

        const loadedBooks = await Promise.all(
          favorites.map(async (favorite) => {
            try {
              return await fetchBookById(favorite.bookId);
            } catch (err) {
              console.error(err);
              return favoriteToBook(favorite);
            }
          })
        );

        if (!active) return;

        setFavoriteBooks(favoriteMap);
        setBooks(loadedBooks);
      } catch (err) {
        console.error(err);

        if (!active) return;

        setError("Ocurrió un error al cargar tus favoritos");
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadFavorites();

    return () => {
      active = false;
    };
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

  const filteredBooks = useMemo(() => {
    const query = normalizeText(activeFilters.query);
    const title = normalizeText(activeFilters.title);
    const author = normalizeText(activeFilters.author);
    const category = normalizeText(activeFilters.category);

    return books.filter((book) => {
      const bookTitle = normalizeText(book.title);
      const bookAuthors = normalizeText(book.authors);
      const bookCategory = normalizeText(book.category);
      const bookEditorial = normalizeText(book.editorialArea);
      const bookKeywords = normalizeText(book.keywords);
      const bookAbstract = normalizeText(book.abstract);

      const matchesQuery =
        !query ||
        [
          bookTitle,
          bookAuthors,
          bookCategory,
          bookEditorial,
          bookKeywords,
          bookAbstract,
        ].some((field) => field.includes(query));

      const matchesTitle = !title || bookTitle.includes(title);
      const matchesAuthor = !author || bookAuthors.includes(author);
      const matchesCategory =
        !category ||
        bookCategory.includes(category) ||
        bookKeywords.includes(category);

      const matchesCitations = book.citations >= activeFilters.minCitations;

      const matchesEditorialCount =
        book.editorialCount >= activeFilters.minEditorialCount;

      return (
        matchesQuery &&
        matchesTitle &&
        matchesAuthor &&
        matchesCategory &&
        matchesCitations &&
        matchesEditorialCount
      );
    });
  }, [books, activeFilters]);

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

    if (wasFavorite) {
      setBooks((prev) => prev.filter((item) => item.id !== book.id));

      if (selectedBook?.id === book.id) {
        setSelectedBook(null);
      }
    }

    try {
      if (wasFavorite) {
        await removeFavorite(user.uid, book.id);
      } else {
        await saveFavorite(user.uid, {
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
        });
      }
    } catch (err) {
      console.error(err);
      setError("No se pudo actualizar el favorito. Intenta nuevamente.");
    }
  };

  const toggleLikedRecommendation = (bookId: string) => {
    setLikedRecommendations((prev) => ({
      ...prev,
      [bookId]: !prev[bookId],
    }));
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    setActiveFilters({
      query: searchInput.trim(),
      title: titleFilter.trim(),
      author: authorFilter.trim(),
      category: categoryFilter.trim(),
      minCitations,
      minEditorialCount,
    });

    setFiltersOpen(false);
    setSelectedBook(null);
  };

  const handleClearFilters = () => {
    setSearchInput("");
    setTitleFilter("");
    setAuthorFilter("");
    setCategoryFilter("");
    setMinCitations(0);
    setMinEditorialCount(0);
    setActiveFilters({
      query: "",
      title: "",
      author: "",
      category: "",
      minCitations: 0,
      minEditorialCount: 0,
    });
    setFiltersOpen(false);
    setSelectedBook(null);
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
          <Image src="/favorites.png" alt="Favoritos" width={42} height={42} />
          <h1>Tus favoritos</h1>
        </div>

        {activeFilters.query ? (
          <p className="search-summary">
            Buscando en favoritos: <strong>{activeFilters.query}</strong>
          </p>
        ) : null}

        <div className="book-grid">
          {authLoading || loading ? (
            <div className="status-message">Cargando favoritos...</div>
          ) : error ? (
            <div className="status-message error">{error}</div>
          ) : filteredBooks.length > 0 ? (
            filteredBooks.map((book) => (
              <BookCard
                key={book.id}
                book={book}
                isFavorite={Boolean(favoriteBooks[book.id])}
                onOpen={setSelectedBook}
                onToggleFavorite={toggleFavorite}
              />
            ))
          ) : books.length === 0 ? (
            <div className="no-results">
              Aún no tienes libros guardados en favoritos.
            </div>
          ) : (
            <div className="no-results">
              No se encontraron favoritos con los filtros actuales.
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