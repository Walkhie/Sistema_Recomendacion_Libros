"use client";

import { FormEvent, useEffect, useState } from "react";
import Image from "next/image";

interface Book {
  id: string;
  title: string;
  edition: string;
  category: string;
  authors: string;
  citations: number;
  editorialCount: number;
  editorialArea: string;
}

function BookCard({ book }: { book: Book }) {
  const [favored, setFavored] = useState(false);

  return (
    <article className="book-card">
      <div className="card-header">
        <div className="card-title-block">
          <h3 className="card-title">{book.title}</h3>
          <p className="card-edition">{book.edition}</p>
        </div>

        <button
          className={`heart-btn ${favored ? "liked" : ""}`}
          onClick={() => setFavored(!favored)}
          aria-label={favored ? "Quitar de favoritos" : "Agregar a favoritos"}
          type="button"
        >
          {favored ? (
            <Image src="/favorite.png" alt="Favored" width={22} height={22} />
          ) : (
            <Image src="/heart.png" alt="Favorite" width={22} height={22} />
          )}
        </button>
      </div>

      <span className="category-pill">{book.category}</span>

      <p className="card-authors">{book.authors}</p>

      <div className="card-footer">
        <p className="card-citations">
          <strong>Citado por {book.citations}</strong>
        </p>
        <p className="card-editorial">
          <strong>Respaldo editorial:</strong> {book.editorialCount} títulos
          publicados en <strong>{book.editorialArea}</strong>
        </p>
      </div>
    </article>
  );
}

export default function HomePage() {
  const [books, setBooks] = useState<Book[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [filtersOpen, setFiltersOpen] = useState(false);

  const [searchInput, setSearchInput] = useState("");
  const [appliedQuery, setAppliedQuery] = useState("");

  const [titleFilter, setTitleFilter] = useState("");
  const [authorFilter, setAuthorFilter] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("");
  const [minCitations, setMinCitations] = useState(0);
  const [minEditorialCount, setMinEditorialCount] = useState(0);

  const loadBooks = async (params?: {
    query?: string;
    title?: string;
    author?: string;
    category?: string;
    min_citations?: number;
    min_editorial_count?: number;
  }) => {
    try {
      setLoading(true);
      setError("");

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

      const url = `http://127.0.0.1:8000/books${
        searchParams.toString() ? `?${searchParams.toString()}` : ""
      }`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error("No se pudieron obtener los libros");
      }

      const data = await response.json();
      setBooks(data);
    } catch (err) {
      console.error(err);
      setError("Ocurrió un error al cargar los libros");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadBooks();
  }, []);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const trimmedQuery = searchInput.trim();
    setAppliedQuery(trimmedQuery);

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
    await loadBooks();
  };

  return (
    <>
      <style>{`
        *, *::before, *::after {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        html, body {
          background: #ffffff;
          color: #111111;
          min-height: 100vh;
          font-family: Arial, Helvetica, sans-serif;
        }

        body {
          overflow-x: hidden;
        }

        .page-shell {
          min-height: 100vh;
          background: #ffffff;
        }

        .navbar {
          background: #ffffff;
          display: grid;
          grid-template-columns: 160px minmax(0, 1fr) 220px;
          align-items: start;
          gap: 20px;
          padding: 20px 28px 14px;
          border-bottom: 1px solid #efefef;
        }

        .nav-logo {
          display: flex;
          align-items: center;
          text-decoration: none;
          width: fit-content;
        }

        .nav-center {
          position: relative;
          display: flex;
          justify-content: center;
          align-items: flex-start;
        }

        .search-form {
          width: min(100%, 980px);
          position: relative;
        }

        .search-wrapper {
          width: 100%;
          height: 58px;
          display: flex;
          align-items: center;
          gap: 14px;
          padding: 0 18px 0 20px;
          background: #f5f5f5;
          border: 1px solid #ececec;
          border-radius: 999px;
        }

        .menu-btn,
        .search-icon-btn {
          background: none;
          border: none;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0;
          opacity: 0.9;
        }

        .search-input {
          flex: 1;
          border: none;
          outline: none;
          background: transparent;
          color: #222;
          font-size: 1rem;
          font-family: Arial, Helvetica, sans-serif;
        }

        .search-input::placeholder {
          color: #7d7d7d;
        }

        .filters-panel {
          position: absolute;
          top: 68px;
          left: 26px;
          width: calc(100% - 52px);
          background: #f8f8f8;
          border: 1px solid #ececec;
          border-radius: 0 0 24px 24px;
          padding: 28px 36px 26px;
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
          z-index: 20;
        }

        .filters-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 18px;
        }

        .filter-row {
          display: grid;
          grid-template-columns: 160px 1fr;
          gap: 18px;
          align-items: center;
        }

        .filter-row label {
          font-size: 0.98rem;
          font-weight: 600;
          color: #3c3c3c;
        }

        .filter-text-input {
          width: 100%;
          border: none;
          border-bottom: 2px solid #b9b9b9;
          background: transparent;
          padding: 8px 2px;
          font-size: 0.96rem;
          outline: none;
        }

        .range-block {
          display: flex;
          align-items: center;
          gap: 14px;
        }

        .range-value {
          min-width: 42px;
          height: 42px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: 999px;
          background: #7b3fe4;
          color: #ffffff;
          font-size: 0.85rem;
          font-weight: 700;
        }

        .range-input {
          width: 100%;
          accent-color: #7b3fe4;
          cursor: pointer;
        }

        .filters-actions {
          margin-top: 10px;
          display: flex;
          justify-content: flex-end;
          gap: 12px;
        }

        .filters-secondary-btn,
        .filters-primary-btn {
          border: none;
          border-radius: 999px;
          padding: 10px 18px;
          font-size: 0.92rem;
          font-weight: 600;
          cursor: pointer;
        }

        .filters-secondary-btn {
          background: #ececec;
          color: #333333;
        }

        .filters-primary-btn {
          background: #7b3fe4;
          color: #ffffff;
        }

        .user-section {
          display: flex;
          justify-content: flex-end;
          align-items: center;
          gap: 10px;
          color: #111111;
          font-size: 0.95rem;
          font-weight: 500;
          padding-top: 8px;
        }

        .main-content {
          max-width: 1280px;
          margin: 0 auto;
          padding: 52px 34px 72px;
          background: #ffffff;
        }

        .page-heading {
          display: flex;
          align-items: center;
          gap: 16px;
          margin-bottom: 42px;
        }

        .page-heading h1 {
          font-size: clamp(2rem, 3vw, 3rem);
          font-weight: 700;
          line-height: 1.1;
          letter-spacing: -0.03em;
          color: #111111;
          font-family: Arial, Helvetica, sans-serif;
        }

        .search-summary {
          margin: -18px 0 30px 52px;
          font-size: 0.95rem;
          color: #6a6a6a;
        }

        .book-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 40px 56px;
        }

        .book-card {
          background: #ffffff;
          border: 1px solid #dcdcdc;
          border-radius: 14px;
          padding: 20px 22px 18px;
          min-height: 210px;
          display: flex;
          flex-direction: column;
          gap: 10px;
          transition: box-shadow 0.2s ease, transform 0.2s ease;
        }

        .book-card:hover {
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
          transform: translateY(-2px);
        }

        .card-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 14px;
        }

        .card-title-block {
          flex: 1;
        }

        .card-title {
          font-size: 0.95rem;
          line-height: 1.35;
          font-weight: 700;
          color: #111111;
          font-family: Arial, Helvetica, sans-serif;
        }

        .card-edition {
          margin-top: 6px;
          font-size: 0.82rem;
          color: #606060;
          font-weight: 400;
        }

        .heart-btn {
          background: none;
          border: none;
          cursor: pointer;
          flex-shrink: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 2px;
          border-radius: 50%;
          transition: transform 0.2s ease;
        }

        .heart-btn:hover {
          transform: scale(1.08);
        }

        .heart-btn.liked {
          animation: pop 0.25s ease;
        }

        @keyframes pop {
          0% { transform: scale(1); }
          50% { transform: scale(1.25); }
          100% { transform: scale(1); }
        }

        .category-pill {
          display: inline-flex;
          align-items: center;
          width: fit-content;
          background: #efe7fb;
          color: #6f53b9;
          border-radius: 999px;
          padding: 5px 12px;
          font-size: 0.76rem;
          font-weight: 500;
        }

        .card-authors {
          font-size: 0.84rem;
          line-height: 1.45;
          color: #2d2d2d;
        }

        .card-footer {
          margin-top: auto;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .card-citations,
        .card-editorial {
          font-size: 0.84rem;
          line-height: 1.4;
          color: #111111;
        }

        .status-message,
        .no-results {
          grid-column: 1 / -1;
          text-align: center;
          padding: 70px 0;
          color: #777;
          font-size: 1rem;
        }

        .status-message.error {
          color: #b42318;
        }

        @media (max-width: 1100px) {
          .navbar {
            grid-template-columns: 130px minmax(0, 1fr) 180px;
          }

          .book-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 28px;
          }
        }

        @media (max-width: 760px) {
          .navbar {
            grid-template-columns: 1fr;
            gap: 16px;
            padding: 18px 18px 12px;
          }

          .nav-logo {
            justify-self: center;
          }

          .user-section {
            justify-content: center;
            padding-top: 0;
          }

          .main-content {
            padding: 30px 18px 50px;
          }

          .page-heading {
            align-items: flex-start;
            margin-bottom: 28px;
          }

          .book-grid {
            grid-template-columns: 1fr;
            gap: 22px;
          }

          .search-wrapper {
            width: 100%;
            height: 54px;
          }

          .filters-panel {
            left: 0;
            width: 100%;
            padding: 22px 18px;
          }

          .filter-row {
            grid-template-columns: 1fr;
            gap: 10px;
          }

          .search-summary {
            margin-left: 0;
          }
        }
      `}</style>

      <div className="page-shell">
        <nav className="navbar">
          <a href="/" className="nav-logo" aria-label="BookMatch">
            <Image
              src="/logoTitulo.png"
              alt="BookMatch"
              width={110}
              height={90}
              priority
              style={{ objectFit: "contain" }}
            />
          </a>

          <div className="nav-center">
            <form className="search-form" onSubmit={handleSubmit}>
              <div className="search-wrapper">
                <button
                  className="menu-btn"
                  aria-label="Abrir filtros"
                  type="button"
                  onClick={() => setFiltersOpen((prev) => !prev)}
                >
                  <Image src="/menu.png" alt="Menú" width={20} height={20} />
                </button>

                <input
                  className="search-input"
                  type="text"
                  placeholder="Buscar"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                />

                <button className="search-icon-btn" aria-label="Buscar" type="submit">
                  <Image src="/search.png" alt="Buscar" width={20} height={20} />
                </button>
              </div>

              {filtersOpen && (
                <div className="filters-panel">
                  <div className="filters-grid">
                    <div className="filter-row">
                      <label htmlFor="filter-title">Título:</label>
                      <input
                        id="filter-title"
                        className="filter-text-input"
                        type="text"
                        value={titleFilter}
                        onChange={(e) => setTitleFilter(e.target.value)}
                      />
                    </div>

                    <div className="filter-row">
                      <label htmlFor="filter-author">Autor:</label>
                      <input
                        id="filter-author"
                        className="filter-text-input"
                        type="text"
                        value={authorFilter}
                        onChange={(e) => setAuthorFilter(e.target.value)}
                      />
                    </div>

                    <div className="filter-row">
                      <label htmlFor="filter-category">Tema:</label>
                      <input
                        id="filter-category"
                        className="filter-text-input"
                        type="text"
                        value={categoryFilter}
                        onChange={(e) => setCategoryFilter(e.target.value)}
                      />
                    </div>

                    <div className="filter-row">
                      <label htmlFor="filter-editorial-count">
                        Respaldo editorial:
                      </label>
                      <div className="range-block">
                        <span className="range-value">{minEditorialCount}</span>
                        <input
                          id="filter-editorial-count"
                          className="range-input"
                          type="range"
                          min="0"
                          max="100"
                          step="1"
                          value={minEditorialCount}
                          onChange={(e) =>
                            setMinEditorialCount(Number(e.target.value))
                          }
                        />
                      </div>
                    </div>

                    <div className="filter-row">
                      <label htmlFor="filter-citations">
                        Citaciones mínimas:
                      </label>
                      <div className="range-block">
                        <span className="range-value">{minCitations}</span>
                        <input
                          id="filter-citations"
                          className="range-input"
                          type="range"
                          min="0"
                          max="100"
                          step="1"
                          value={minCitations}
                          onChange={(e) =>
                            setMinCitations(Number(e.target.value))
                          }
                        />
                      </div>
                    </div>
                  </div>

                  <div className="filters-actions">
                    <button
                      type="button"
                      className="filters-secondary-btn"
                      onClick={handleClearFilters}
                    >
                      Limpiar
                    </button>
                    <button type="submit" className="filters-primary-btn">
                      Aplicar filtros
                    </button>
                  </div>
                </div>
              )}
            </form>
          </div>

          <div className="user-section">
            <Image src="/user.png" alt="Usuario" width={48} height={48} />
            <span>Nombre del Usuario</span>
          </div>
        </nav>

        <main className="main-content">
          <div className="page-heading">
            <Image src="/home.png" alt="Inicio" width={34} height={34} />
            <h1>Página Principal: Recursos de tu interés</h1>
          </div>

          {appliedQuery && (
            <p className="search-summary">
              Resultados para: <strong>{appliedQuery}</strong>
            </p>
          )}

          <div className="book-grid">
            {loading ? (
              <div className="status-message">Cargando libros...</div>
            ) : error ? (
              <div className="status-message error">{error}</div>
            ) : books.length > 0 ? (
              books.map((book) => <BookCard key={book.id} book={book} />)
            ) : (
              <div className="no-results">
                No se encontraron libros con los filtros actuales.
              </div>
            )}
          </div>
        </main>
      </div>
    </>
  );
}