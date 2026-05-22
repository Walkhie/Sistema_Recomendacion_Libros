"use client";

import type { MouseEvent } from "react";
import Image from "next/image";

import type { Book } from "../types/book";
import type { RecommendationReaction } from "@/lib/userStore";

interface BookDetailModalProps {
  book: Book | null;
  sourceBook?: Book | null;
  isOpen: boolean;
  isFavorite: boolean;
  isLiked: boolean;
  isDisliked: boolean;
  onClose: () => void;
  onToggleFavorite: (book: Book) => void | Promise<void>;
  onToggleReaction: (
    book: Book,
    reaction: RecommendationReaction
  ) => void | Promise<void>;
}

function formatValue(value?: string | number) {
  if (value === undefined || value === null) return "No disponible";

  const text = String(value).trim();
  return text.length > 0 ? text : "No disponible";
}

export default function BookDetailModal({
  book,
  sourceBook,
  isOpen,
  isFavorite,
  isLiked,
  isDisliked,
  onClose,
  onToggleFavorite,
  onToggleReaction,
}: BookDetailModalProps) {
  if (!isOpen || !book) return null;

  const doi = formatValue(book.doi);
  const abstract = formatValue(book.abstract);
  const institution = formatValue(book.institution);
  const language = formatValue(book.language);
  const editorial = formatValue(book.editorial || book.editorialArea);
  const year = formatValue(book.year || book.edition);
  const authors = formatValue(book.authors);
  const category = formatValue(book.category);
  const showFeedback = Boolean(sourceBook);

  const isDoiUrl =
    typeof book.doi === "string" &&
    (book.doi.startsWith("http://") || book.doi.startsWith("https://"));

  const handleOverlayClick = (event: MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      onClose();
    }
  };

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div
        className="book-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="book-modal-title"
      >
        <div className="book-modal__header">
          <div className="book-modal__heading">
            <h2 id="book-modal-title" className="book-modal__title">
              {book.title}
            </h2>

            <p className="book-modal__authors">{authors}</p>

            <p className="book-modal__edition">{year}</p>

            <span className="category-pill" title={category}>
              {category}
            </span>
          </div>

          <button
            type="button"
            className={`heart-btn book-modal__favorite-btn ${
              isFavorite ? "liked" : ""
            }`}
            onClick={() => onToggleFavorite(book)}
            aria-label={
              isFavorite ? "Quitar de favoritos" : "Agregar a favoritos"
            }
          >
            {isFavorite ? (
              <Image src="/favorite.png" alt="Favored" width={22} height={22} />
            ) : (
              <Image src="/heart.png" alt="Favorite" width={22} height={22} />
            )}
          </button>
        </div>

        <div className="book-modal__grid">
          <div className="book-modal__detail">
            <span className="book-modal__label">Citaciones</span>
            <span className="book-modal__value">{book.citations}</span>
          </div>

          <div className="book-modal__detail">
            <span className="book-modal__label">Respaldo editorial</span>
            <span className="book-modal__value">
              {book.editorialCount} títulos en {book.editorialArea}
            </span>
          </div>

          <div className="book-modal__detail">
            <span className="book-modal__label">Editorial</span>
            <span className="book-modal__value">{editorial}</span>
          </div>

          <div className="book-modal__detail">
            <span className="book-modal__label">Idioma</span>
            <span className="book-modal__value">{language}</span>
          </div>

          <div className="book-modal__detail">
            <span className="book-modal__label">Institución</span>
            <span className="book-modal__value">{institution}</span>
          </div>

          <div className="book-modal__detail">
            <span className="book-modal__label">DOI</span>
            <span className="book-modal__value">
              {isDoiUrl ? (
                <a
                  href={book.doi}
                  target="_blank"
                  rel="noreferrer"
                  className="book-modal__link"
                >
                  {book.doi}
                </a>
              ) : (
                doi
              )}
            </span>
          </div>
        </div>

        <div className="book-modal__section">
          <h3 className="book-modal__section-title">Resumen</h3>
          <p className="book-modal__text">{abstract}</p>
        </div>

        {showFeedback ? (
          <div className="book-modal__feedback">
            <span className="book-modal__feedback-text">
              ¿Te sirvió esta recomendación?
            </span>

            <div className="book-modal__feedback-actions">
              <button
                type="button"
                className={`modal-icon-btn ${isLiked ? "liked" : ""}`}
                onClick={() => onToggleReaction(book, "like")}
                aria-label={
                  isLiked
                    ? "Quitar like de la recomendación"
                    : "Dar like a la recomendación"
                }
              >
                {isLiked ? (
                  <Image src="/liked.png" alt="Liked" width={30} height={30} />
                ) : (
                  <Image src="/like.png" alt="Like" width={30} height={30} />
                )}
              </button>

              <button
                type="button"
                className={`modal-icon-btn ${isDisliked ? "disliked" : ""}`}
                onClick={() => onToggleReaction(book, "dislike")}
                aria-label={
                  isDisliked
                    ? "Quitar dislike de la recomendación"
                    : "Dar dislike a la recomendación"
                }
              >
                {isDisliked ? (
                  <Image
                    src="/disliked.png"
                    alt="Disliked"
                    width={30}
                    height={30}
                  />
                ) : (
                  <Image
                    src="/dislike.png"
                    alt="Dislike"
                    width={30}
                    height={30}
                  />
                )}
              </button>
            </div>
          </div>
        ) : null}

        <div className="book-modal__actions">
          <button
            type="button"
            className="book-modal__close-btn"
            onClick={onClose}
          >
            Seguir explorando
          </button>
        </div>
      </div>
    </div>
  );
}