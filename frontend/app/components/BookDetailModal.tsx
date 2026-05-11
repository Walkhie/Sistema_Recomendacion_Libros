"use client";

import Image from "next/image";

import type { Book } from "../types/book";

interface BookDetailModalProps {
  book: Book | null;
  isOpen: boolean;
  isFavorite: boolean;
  isLiked: boolean;
  onClose: () => void;
  onToggleFavorite: (book: Book) => void | Promise<void>;
  onToggleLiked: (bookId: string) => void;
}

function formatValue(value?: string | number) {
  if (value === undefined || value === null) return "No disponible";

  const text = String(value).trim();
  return text.length > 0 ? text : "No disponible";
}

export default function BookDetailModal({
  book,
  isOpen,
  isFavorite,
  isLiked,
  onClose,
  onToggleFavorite,
  onToggleLiked,
}: BookDetailModalProps) {
  if (!isOpen || !book) return null;

  const doi = formatValue(book.doi);
  const abstract = formatValue(book.abstract);
  const keywords = formatValue(book.keywords);
  const institution = formatValue(book.institution);
  const language = formatValue(book.language);
  const editorial = formatValue(book.editorial || book.editorialArea);
  const year = formatValue(book.year || book.edition);

  const isDoiUrl =
    typeof book.doi === "string" &&
    (book.doi.startsWith("http://") || book.doi.startsWith("https://"));

  return (
    <div className="modal-overlay">
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
            <p className="book-modal__edition">{year}</p>
          </div>

          <button
            type="button"
            className={`modal-icon-btn ${isFavorite ? "liked" : ""}`}
            onClick={() => onToggleFavorite(book)}
            aria-label={
              isFavorite ? "Quitar de favoritos" : "Agregar a favoritos"
            }
          >
            {isFavorite ? (
              <Image src="/favorite.png" alt="Favored" width={28} height={28} />
            ) : (
              <Image src="/heart.png" alt="Favorite" width={28} height={28} />
            )}
          </button>
        </div>

        <div className="book-modal__meta">
          <span className="category-pill" title={book.category}>
            {book.category}
          </span>
          <p className="book-modal__authors">{book.authors}</p>
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

        <div className="book-modal__section">
          <h3 className="book-modal__section-title">Palabras clave</h3>
          <p className="book-modal__text">{keywords}</p>
        </div>

        <div className="book-modal__feedback">
          <span className="book-modal__feedback-text">
            ¿Quedó satisfecho con la recomendación?
          </span>

          <button
            type="button"
            className={`modal-icon-btn ${isLiked ? "liked" : ""}`}
            onClick={() => onToggleLiked(book.id)}
            aria-label={
              isLiked
                ? "Quitar calificación positiva"
                : "Calificar recomendación positivamente"
            }
          >
            {isLiked ? (
              <Image src="/liked.png" alt="Liked" width={30} height={30} />
            ) : (
              <Image src="/like.png" alt="Like" width={30} height={30} />
            )}
          </button>
        </div>

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