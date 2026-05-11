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
  clearRecommendationReaction,
  getUserFavorites,
  getUserRecommendationFeedback,
  removeFavorite,
  saveFavorite,
  setRecommendationReaction,
  type BookSummary,
  type FavoriteBook,
  type RecommendationFeedback,
  type RecommendationReaction,
} from "@/lib/userStore";

const API_BASE_URL = "http://127.0.0.1:8000";
const RECOMMENDATIONS_PER_SOURCE = 6;
const RECOMMENDATION_FETCH_LIMIT = 18;
const MAX_RECOMMENDATION_SEEDS_WHEN_USING_LIKES = 5;

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

type RecommendationSourceType = "favorite" | "liked_recommendation";

type RecommendationSeed = {
  seedBook: Book;
  sourceType: RecommendationSourceType;
};

type RecommendationGroup = RecommendationSeed & {
  recommendations: Book[];
};

type SelectedBookContext = {
  book: Book;
  sourceBook?: Book | null;
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

function bookSummaryToBook(book: BookSummary): Book {
  return {
    id: book.id,
    title: book.title,
    edition: book.edition ?? book.year ?? "",
    category: book.category ?? "",
    authors: book.authors ?? "",
    citations: book.citations ?? 0,
    editorialCount: book.editorialCount ?? 0,
    editorialArea: book.editorialArea ?? "",
    year: book.year ?? book.edition ?? "",
    editorial: book.editorial ?? book.editorialArea ?? "",
    doi: book.doi ?? "",
    abstract: book.abstract ?? "",
    keywords: book.keywords ?? "",
    language: book.language ?? "",
    institution: book.institution ?? "",
    matchMethod: "",
    openAlexId: "",
    editorialScore: 0,
    citationScore: 0,
  };
}

function bookToFavoritePayload(book: Book): BookSummary {
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

function buildReactionKey(sourceBookId: string, bookId: string) {
  return `${sourceBookId}__${bookId}`;
}

function feedbackToReactionMap(feedback: RecommendationFeedback[]) {
  return feedback.reduce<Record<string, RecommendationReaction>>((acc, item) => {
    if (item.sourceBookId && item.bookId) {
      acc[buildReactionKey(item.sourceBookId, item.bookId)] = item.reaction;
    }

    return acc;
  }, {});
}

function buildRecommendationSeeds(
  favorites: FavoriteBook[],
  feedback: RecommendationFeedback[]
): RecommendationSeed[] {
  const seeds: RecommendationSeed[] = [];
  const usedSeedIds = new Set<string>();

  favorites.forEach((favorite) => {
    const seedBook = favoriteToBook(favorite);

    if (!seedBook.id || usedSeedIds.has(seedBook.id)) return;

    seeds.push({
      seedBook,
      sourceType: "favorite",
    });

    usedSeedIds.add(seedBook.id);
  });

  if (favorites.length >= MAX_RECOMMENDATION_SEEDS_WHEN_USING_LIKES) {
    return seeds;
  }

  for (const item of feedback) {
    if (seeds.length >= MAX_RECOMMENDATION_SEEDS_WHEN_USING_LIKES) break;
    if (item.reaction !== "like") continue;
    if (!item.recommendedBook?.id) continue;
    if (usedSeedIds.has(item.recommendedBook.id)) continue;

    const seedBook = bookSummaryToBook(item.recommendedBook);

    if (!seedBook.id) continue;

    seeds.push({
      seedBook,
      sourceType: "liked_recommendation",
    });

    usedSeedIds.add(seedBook.id);
  }

  return seeds;
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

  const [selectedBookContext, setSelectedBookContext] =
    useState<SelectedBookContext | null>(null);

  const [favoriteBooks, setFavoriteBooks] = useState<Record<string, boolean>>(
    {}
  );

  const [recommendationFeedback, setRecommendationFeedback] = useState<
    Record<string, RecommendationReaction>
  >({});

  const fetchBookById = async (bookId: string): Promise<Book> => {
    const response = await fetch(`${API_BASE_URL}/books/${bookId}`);

    if (!response.ok) {
      throw new Error(`No se pudo obtener el libro ${bookId}`);
    }

    return response.json();
  };

  const fetchBookByIdSafely = async (bookId: string) => {
    try {
      return await fetchBookById(bookId);
    } catch (err) {
      console.error(err);
      return null;
    }
  };

  const loadRecommendationsForSeed = async (
    seed: RecommendationSeed,
    feedback: RecommendationFeedback[]
  ): Promise<RecommendationGroup> => {
    const { seedBook } = seed;

    try {
      const response = await fetch(
        `${API_BASE_URL}/books/${seedBook.id}/recommendations?top_n=${RECOMMENDATION_FETCH_LIMIT}`
      );

      if (!response.ok) {
        throw new Error(
          `No se pudieron cargar recomendaciones para ${seedBook.title}`
        );
      }

      const data: RecommendationResponse = await response.json();

      const dislikedBookIdsForThisSource = new Set(
        feedback
          .filter(
            (item) =>
              item.sourceBookId === seedBook.id && item.reaction === "dislike"
          )
          .map((item) => item.bookId)
      );

      const recommendedIds = data.recommendations
        .map((item) => item["Código del libro"])
        .filter((bookId) => Boolean(bookId))
        .filter((bookId) => !dislikedBookIdsForThisSource.has(bookId));

      const uniqueRecommendedIds = Array.from(new Set(recommendedIds)).slice(
        0,
        RECOMMENDATIONS_PER_SOURCE
      );

      const recommendations = (
        await Promise.all(uniqueRecommendedIds.map(fetchBookByIdSafely))
      ).filter((book): book is Book => Boolean(book));

      return {
        ...seed,
        recommendations,
      };
    } catch (err) {
      console.error(err);

      return {
        ...seed,
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
        setRecommendationFeedback({});
        setRecommendationGroups([]);
        return;
      }

      const [favorites, feedback] = await Promise.all([
        getUserFavorites(user.uid),
        getUserRecommendationFeedback(user.uid),
      ]);

      const favoriteMap = favorites.reduce<Record<string, boolean>>(
        (acc, favorite) => {
          acc[favorite.bookId] = true;
          return acc;
        },
        {}
      );

      setFavoriteBooks(favoriteMap);
      setRecommendationFeedback(feedbackToReactionMap(feedback));

      const seeds = buildRecommendationSeeds(favorites, feedback);

      if (seeds.length === 0) {
        setRecommendationGroups([]);
        return;
      }

      const groups = await Promise.all(
        seeds.map((seed) => loadRecommendationsForSeed(seed, feedback))
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

    if (selectedBookContext) {
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, [selectedBookContext]);

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

  const toggleRecommendationReaction = async (
    book: Book,
    reaction: RecommendationReaction,
    sourceBook?: Book | null
  ) => {
    if (!user) {
      router.push("/login");
      return;
    }

    if (!sourceBook) return;

    const reactionKey = buildReactionKey(sourceBook.id, book.id);
    const previousReaction = recommendationFeedback[reactionKey];
    const nextReaction = previousReaction === reaction ? undefined : reaction;

    setRecommendationFeedback((prev) => {
      const next = { ...prev };

      if (nextReaction) {
        next[reactionKey] = nextReaction;
      } else {
        delete next[reactionKey];
      }

      return next;
    });

    if (nextReaction === "dislike") {
      setRecommendationGroups((prev) =>
        prev.map((group) => {
          if (group.seedBook.id !== sourceBook.id) return group;

          return {
            ...group,
            recommendations: group.recommendations.filter(
              (recommendation) => recommendation.id !== book.id
            ),
          };
        })
      );
    }

    try {
      if (!nextReaction) {
        await clearRecommendationReaction(user.uid, sourceBook.id, book.id);
      } else {
        await setRecommendationReaction(user.uid, {
          book: bookToFavoritePayload(book),
          sourceBook: bookToFavoritePayload(sourceBook),
          reaction: nextReaction,
        });
      }
    } catch (err) {
      console.error(err);

      setRecommendationFeedback((prev) => {
        const next = { ...prev };

        if (previousReaction) {
          next[reactionKey] = previousReaction;
        } else {
          delete next[reactionKey];
        }

        return next;
      });
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const trimmedQuery = searchInput.trim();
    setAppliedQuery(trimmedQuery);
    setSelectedBookContext(null);

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
    setSelectedBookContext(null);

    await loadPersonalizedRecommendations();
  };

  const selectedBook = selectedBookContext?.book ?? null;
  const selectedSourceBook = selectedBookContext?.sourceBook ?? null;

  const selectedReactionKey =
    selectedBook && selectedSourceBook
      ? buildReactionKey(selectedSourceBook.id, selectedBook.id)
      : "";

  const selectedReaction = selectedReactionKey
    ? recommendationFeedback[selectedReactionKey]
    : undefined;

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
            Recomendaciones generadas a partir de tus favoritos y de las
            recomendaciones que te gustaron.
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
                  onOpen={(selected) =>
                    setSelectedBookContext({ book: selected })
                  }
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
                  key={`${group.sourceType}-${group.seedBook.id}`}
                  className="home-recommendation-page"
                >
                  <div className="home-recommendation-header">
                    <span>
                      {group.sourceType === "favorite"
                        ? "Basado en tu favorito"
                        : "Basado en una recomendación que te gustó"}
                    </span>
                    <h2>{group.seedBook.title}</h2>
                  </div>

                  <div className="home-recommendation-grid">
                    {group.recommendations.map((book) => (
                      <BookCard
                        key={`${group.seedBook.id}-${book.id}`}
                        book={book}
                        isFavorite={Boolean(favoriteBooks[book.id])}
                        onOpen={(selected) =>
                          setSelectedBookContext({
                            book: selected,
                            sourceBook: group.seedBook,
                          })
                        }
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
                otros favoritos o en recomendaciones que te gustaron.
              </p>
            ) : null}
          </>
        ) : user ? (
          <div className="no-results">
            Aún no tienes favoritos ni recomendaciones marcadas con like para
            generar recomendaciones personalizadas. Selecciona libros desde el
            onboarding o marca libros con el corazón.
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
        sourceBook={selectedSourceBook}
        isOpen={Boolean(selectedBook)}
        isFavorite={selectedBook ? Boolean(favoriteBooks[selectedBook.id]) : false}
        isLiked={selectedReaction === "like"}
        isDisliked={selectedReaction === "dislike"}
        onClose={() => setSelectedBookContext(null)}
        onToggleFavorite={toggleFavorite}
        onToggleReaction={(book, reaction) =>
          toggleRecommendationReaction(book, reaction, selectedSourceBook)
        }
      />
    </div>
  );
}