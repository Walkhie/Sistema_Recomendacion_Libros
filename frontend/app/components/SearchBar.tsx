"use client";

import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";

import { useAuth } from "@/context/AuthContext";
import { logoutUser } from "@/lib/auth";
import { getUserProfile } from "@/lib/userStore";

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

type BasicProfile = {
  fullName?: string;
  firstName?: string;
  lastName?: string;
};

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
  const router = useRouter();
  const { user, loading } = useAuth();

  const [menuOpen, setMenuOpen] = useState(false);
  const [profileName, setProfileName] = useState("");

  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;

    async function loadProfile() {
      if (!user) {
        setProfileName("");
        return;
      }

      try {
        const profile = (await getUserProfile(user.uid)) as BasicProfile | null;

        if (!active) return;

        const firestoreName =
          profile?.fullName?.trim() ||
          `${profile?.firstName ?? ""} ${profile?.lastName ?? ""}`.trim();

        const firebaseName = user.displayName?.trim() || "";
        const emailName = user.email?.split("@")[0] ?? "";

        setProfileName(firestoreName || firebaseName || emailName);
      } catch (error) {
        console.error(error);
        if (!active) return;

        const fallbackName =
          user.displayName?.trim() || user.email?.split("@")[0] || "";
        setProfileName(fallbackName);
      }
    }

    loadProfile();

    return () => {
      active = false;
    };
  }, [user]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (!menuRef.current) return;

      if (!menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const displayName = useMemo(() => {
    if (loading) return "Cargando...";
    if (!user) return "Inicia sesión!";
    return profileName || "Usuario";
  }, [loading, user, profileName]);

  const handleUserClick = () => {
    if (loading) return;

    if (!user) {
      router.push("/login");
      return;
    }

    setMenuOpen((prev) => !prev);
  };

  const handleGoToFavorites = () => {
    setMenuOpen(false);
    router.push("/favoritos");
  };

  const handleLogout = async () => {
    try {
      setMenuOpen(false);
      await logoutUser();
    } catch (error) {
      console.error(error);
    }
  };

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

      <div className="user-menu-wrapper" ref={menuRef}>
        <button
          type="button"
          className="user-section user-section-btn"
          onClick={handleUserClick}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
        >
          <Image src="/user.png" alt="Usuario" width={48} height={48} />
          <span>{displayName}</span>
        </button>

        {user && menuOpen && (
          <div className="user-dropdown" role="menu">
            <button
              type="button"
              className="user-dropdown-item"
              onClick={handleGoToFavorites}
            >
              Favoritos
            </button>

            <button
              type="button"
              className="user-dropdown-item user-dropdown-item--danger"
              onClick={handleLogout}
            >
              Cerrar sesión
            </button>
          </div>
        )}
      </div>
    </nav>
  );
}