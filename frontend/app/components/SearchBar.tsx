"use client";

import type { FormEvent } from "react";
import Image from "next/image";

interface SearchBarProps {
  filtersOpen: boolean;
  searchInput: string;
  titleFilter: string;
  authorFilter: string;
  categoryFilter: string;
  minCitations: number;
  minEditorialCount: number;
  onToggleFilters: () => void;
  onSearchInputChange: (value: string) => void;
  onTitleFilterChange: (value: string) => void;
  onAuthorFilterChange: (value: string) => void;
  onCategoryFilterChange: (value: string) => void;
  onMinCitationsChange: (value: number) => void;
  onMinEditorialCountChange: (value: number) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  onClearFilters: () => void;
}

export default function SearchBar({
  filtersOpen,
  searchInput,
  titleFilter,
  authorFilter,
  categoryFilter,
  minCitations,
  minEditorialCount,
  onToggleFilters,
  onSearchInputChange,
  onTitleFilterChange,
  onAuthorFilterChange,
  onCategoryFilterChange,
  onMinCitationsChange,
  onMinEditorialCountChange,
  onSubmit,
  onClearFilters,
}: SearchBarProps) {
  return (
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
        <form className="search-form" onSubmit={onSubmit}>
          <div className="search-wrapper">
            <button
              className="menu-btn"
              aria-label="Abrir filtros"
              type="button"
              onClick={onToggleFilters}
            >
              <Image src="/menu.png" alt="Menú" width={20} height={20} />
            </button>

            <input
              className="search-input"
              type="text"
              placeholder="Buscar"
              value={searchInput}
              onChange={(e) => onSearchInputChange(e.target.value)}
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
                    onChange={(e) => onTitleFilterChange(e.target.value)}
                  />
                </div>

                <div className="filter-row">
                  <label htmlFor="filter-author">Autor:</label>
                  <input
                    id="filter-author"
                    className="filter-text-input"
                    type="text"
                    value={authorFilter}
                    onChange={(e) => onAuthorFilterChange(e.target.value)}
                  />
                </div>

                <div className="filter-row">
                  <label htmlFor="filter-category">Tema:</label>
                  <input
                    id="filter-category"
                    className="filter-text-input"
                    type="text"
                    value={categoryFilter}
                    onChange={(e) => onCategoryFilterChange(e.target.value)}
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
                        onMinEditorialCountChange(Number(e.target.value))
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
                        onMinCitationsChange(Number(e.target.value))
                      }
                    />
                  </div>
                </div>
              </div>

              <div className="filters-actions">
                <button
                  type="button"
                  className="filters-secondary-btn"
                  onClick={onClearFilters}
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
  );
}