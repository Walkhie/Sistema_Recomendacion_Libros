"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";

import { getPreferences, savePreferences } from "@/lib/userStore";
import { useAuth } from "@/context/AuthContext";
import type { Book } from "@/app/types/book";

const API_URL = "http://127.0.0.1:8000";
const TOP_N_BOOKS = 72;
const BOOKS_PER_PAGE = 36;

function normalizeText(value?: string | number) {
  return String(value ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
}

function chunkBooks(books: Book[], size: number) {
  const chunks: Book[][] = [];

  for (let i = 0; i < books.length; i += size) {
    chunks.push(books.slice(i, i + size));
  }

  return chunks;
}

export default function OnboardingLibrosPage() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();

  const [books, setBooks] = useState<Book[]>([]);
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [selectedBookIds, setSelectedBookIds] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [loadingBooks, setLoadingBooks] = useState(true);

  const displayName = useMemo(() => {
    if (user?.displayName?.trim()) return user.displayName;
    if (user?.email) return user.email.split("@")[0];
    return "Nombre del Usuario";
  }, [user]);

  const bookPages = useMemo(() => chunkBooks(books, BOOKS_PER_PAGE), [books]);

  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    }
  }, [authLoading, user, router]);

  useEffect(() => {
    let active = true;

    async function loadBooksByTopics() {
      if (authLoading) return;

      if (!user) {
        setLoadingBooks(false);
        return;
      }

      try {
        setLoadingBooks(true);
        setError("");

        const preferences = await getPreferences(user.uid);
        const topics = preferences?.topics ?? [];

        if (!topics.length) {
          router.replace("/onboarding/temas");
          return;
        }

        const normalizedTopics = new Set(topics.map(normalizeText));
        setSelectedTopics(topics);

        const params = new URLSearchParams({
          topics: topics.join(","),
          top_n: String(TOP_N_BOOKS),
        });

        const response = await fetch(`${API_URL}/books/top-by-topics?${params}`);

        if (!response.ok) {
          throw new Error("No se pudieron cargar los libros recomendados.");
        }

        const data: Book[] = await response.json();

        const onlySelectedTopicBooks = data
          .filter((book) => normalizedTopics.has(normalizeText(book.category)))
          .sort((a, b) => {
            const citationDiff = (b.citations ?? 0) - (a.citations ?? 0);
            if (citationDiff !== 0) return citationDiff;

            const editorialDiff =
              (b.editorialCount ?? 0) - (a.editorialCount ?? 0);
            if (editorialDiff !== 0) return editorialDiff;

            return a.title.localeCompare(b.title);
          });

        if (!active) return;

        setBooks(onlySelectedTopicBooks);
      } catch (err) {
        console.error(err);

        if (active) {
          setError("No se pudieron cargar los libros recomendados.");
        }
      } finally {
        if (active) {
          setLoadingBooks(false);
        }
      }
    }

    loadBooksByTopics();

    return () => {
      active = false;
    };
  }, [authLoading, user, router]);

  const toggleBook = (bookId: string) => {
    setSelectedBookIds((prev) =>
      prev.includes(bookId)
        ? prev.filter((id) => id !== bookId)
        : [...prev, bookId]
    );
  };

  const handleSubmit = async () => {
    if (!user) {
      router.replace("/login");
      return;
    }

    if (selectedBookIds.length === 0) {
      setError("Selecciona al menos un libro.");
      return;
    }

    try {
      setSaving(true);
      setError("");

      await savePreferences(user.uid, {
        favoriteSeedBookIds: selectedBookIds,
        onboardingCompleted: true,
      });

      router.replace("/");
    } catch (err) {
      console.error(err);
      setError("No se pudieron guardar los libros seleccionados.");
    } finally {
      setSaving(false);
    }
  };

  if (authLoading || loadingBooks || !user) {
    return <main className="auth-flow-page">Cargando...</main>;
  }

  return (
    <main className="auth-flow-page">
      <section className="auth-flow-card auth-flow-card--books">
        <button
          type="button"
          className="auth-flow-back"
          onClick={() => router.push("/onboarding/temas")}
          aria-label="Volver"
        >
          ←
        </button>

        <div className="auth-flow-logo">
          <Image
            src="/logoTitulo.png"
            alt="BookMatch"
            width={126}
            height={126}
            className="auth-flow-logo-image"
            priority
          />
        </div>

        <div className="auth-flow-content auth-flow-content--books">
          <div className="auth-flow-user">
            <span className="auth-flow-user-icon">
              <Image src="/user.png" alt="Usuario" width={24} height={24} />
            </span>
            <span>{displayName}</span>
          </div>

          <h1 className="auth-flow-section-title">
            Selecciona libros recomendados para ti
          </h1>

          <p className="auth-flow-section-copy auth-flow-section-copy--books">
            Estos son los <strong>Top {books.length} libros</strong> asociados
            únicamente a tus temas seleccionados, priorizados por número de
            citaciones. Desliza horizontalmente para ver más grupos.
          </p>

          <div className="selected-topic-row selected-topic-row--books">
            {selectedTopics.map((topic) => (
              <span key={topic} className="selected-topic-pill">
                {topic}
              </span>
            ))}
          </div>

          {books.length > 0 ? (
            <div className="onboarding-books-slider">
              {bookPages.map((pageBooks, pageIndex) => (
                <section
                  key={pageIndex}
                  className="onboarding-books-page"
                  aria-label={`Página ${pageIndex + 1} de libros recomendados`}
                >
                  {pageBooks.map((book) => {
                    const active = selectedBookIds.includes(book.id);

                    return (
                      <button
                        key={book.id}
                        type="button"
                        className={`onboarding-book-card ${
                          active ? "onboarding-book-card--selected" : ""
                        }`}
                        onClick={() => toggleBook(book.id)}
                        title={book.title}
                      >
                        <div className="onboarding-book-card__main">
                          <h2 className="onboarding-book-card__title">
                            {book.title}
                          </h2>

                          <p className="onboarding-book-card__authors">
                            {book.authors || "Autor desconocido"}
                          </p>

                          <p className="onboarding-book-card__year">
                            {book.year || book.edition || "Sin año"}
                          </p>

                          <span className="onboarding-book-card__category">
                            {book.category || "General"}
                          </span>
                        </div>
                        <div className="onboarding-book-card__footer">
                          <p>
                            <strong>Citado por {book.citations ?? 0}</strong>
                          </p>
                          <p>Respaldo editorial: {book.editorialCount ?? 0}</p>
                        </div>
                      </button>
                    );
                  })}
                </section>
              ))}
            </div>
          ) : (
            <p className="auth-form-error">
              No se encontraron libros para los temas seleccionados.
            </p>
          )}

          {bookPages.length > 1 ? (
            <p className="onboarding-scroll-hint">
              Desliza horizontalmente para ver más recomendaciones.
            </p>
          ) : null}

          {error ? <p className="auth-form-error">{error}</p> : null}

          <div className="auth-submit-block auth-submit-block--books">
            <button
              type="button"
              className="auth-primary-btn"
              onClick={handleSubmit}
              disabled={saving}
            >
              {saving ? "Guardando..." : "Finalizar"}
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}