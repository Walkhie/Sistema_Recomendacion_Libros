"use client";

import type { KeyboardEvent } from "react";
import Image from "next/image";

import type { Book } from "../types/book";

interface BookCardProps {
  book: Book;
  isFavorite: boolean;
  onOpen: (book: Book) => void;
  onToggleFavorite: (book: Book) => void | Promise<void>;
}

export default function BookCard({
  book,
  isFavorite,
  onOpen,
  onToggleFavorite,
}: BookCardProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onOpen(book);
    }
  };

  const displayYear = book.year || book.edition || "Sin año";
  const displayAuthors = book.authors || "Autor desconocido";
  const displayCategory = book.category || "General";

  return (
    <article
      className="book-card book-card--interactive"
      role="button"
      tabIndex={0}
      onClick={() => onOpen(book)}
      onKeyDown={handleKeyDown}
      aria-label={`Ver detalles de ${book.title}`}
    >
      <div className="card-top">
        <div className="card-header">
          <div className="card-title-block">
            <h3 className="card-title" title={book.title}>
              {book.title}
            </h3>
          </div>

          <button
            className={`heart-btn ${isFavorite ? "liked" : ""}`}
            onClick={(event) => {
              event.stopPropagation();
              onToggleFavorite(book);
            }}
            aria-label={
              isFavorite ? "Quitar de favoritos" : "Agregar a favoritos"
            }
            type="button"
          >
            {isFavorite ? (
              <Image src="/favorite.png" alt="Favored" width={22} height={22} />
            ) : (
              <Image src="/heart.png" alt="Favorite" width={22} height={22} />
            )}
          </button>
        </div>

        <div className="card-meta">
          <p className="card-authors" title={displayAuthors}>
            {displayAuthors}
          </p>

          <p className="card-edition">{displayYear}</p>

          <span className="category-pill" title={displayCategory}>
            {displayCategory}
          </span>
        </div>
      </div>

      <div className="card-footer">
        <p className="card-citations">
          <strong>Citado por {book.citations}</strong>
        </p>

        <p
          className="card-editorial"
          title={`Respaldo editorial: ${book.editorialCount} títulos publicados en ${book.editorialArea}`}
        >
          <strong>Respaldo editorial:</strong> {book.editorialCount} títulos
          publicados en <strong>{book.editorialArea}</strong>
        </p>
      </div>
    </article>
  );
}