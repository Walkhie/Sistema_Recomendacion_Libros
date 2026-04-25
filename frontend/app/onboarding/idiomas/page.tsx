"use client";

import { useMemo, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { savePreferences } from "@/lib/userStore";
import { useAuth } from "@/context/AuthContext";

const LANGUAGE_OPTIONS = [
  ["es", "Español"],
  ["en", "Inglés"],
  ["pt", "Portugués"],
  ["fr", "Francés"],
] as const;

export default function IdiomasPage() {
  const router = useRouter();
  const { user, loading } = useAuth();

  const [languages, setLanguages] = useState<string[]>(["es"]);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const displayName = useMemo(() => {
    if (user?.displayName?.trim()) return user.displayName;
    if (user?.email) return user.email.split("@")[0];
    return "Nombre del Usuario";
  }, [user]);

  const toggleLanguage = (lang: string) => {
    setLanguages((prev) =>
      prev.includes(lang)
        ? prev.filter((item) => item !== lang)
        : [...prev, lang]
    );
  };

  const handleSubmit = async () => {
    if (!user) {
      setError("Debes iniciar sesión.");
      return;
    }

    if (languages.length === 0) {
      setError("Selecciona al menos un idioma.");
      return;
    }

    try {
      setSaving(true);
      setError("");

      await savePreferences(user.uid, {
        languages,
        topics: [],
        onboardingCompleted: false,
      });

      router.push("/onboarding/temas");
    } catch (err) {
      console.error(err);
      setError("No se pudieron guardar los idiomas.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <main className="auth-flow-page">Cargando...</main>;
  }

  return (
    <main className="auth-flow-page">
      <section className="auth-flow-card">
        <button
          type="button"
          className="auth-flow-back"
          onClick={() => router.push("/registro")}
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

        <div className="auth-flow-content">
          <div className="auth-flow-user">
            <span className="auth-flow-user-icon">
              <Image src="/user.png" alt="Usuario" width={24} height={24} />
            </span>
            <span>{displayName}</span>
          </div>

          <h1 className="auth-flow-section-title">
            ¿En qué idioma te gustaría leer tus libros?
          </h1>

          <p className="auth-flow-section-copy">
            Esto nos ayuda a recomendarte libros que estén en tu idioma de preferencia.
            <strong> Podrás cambiar esta configuración cuando quieras.</strong>
          </p>

          <div className="language-grid">
            {LANGUAGE_OPTIONS.map(([value, label]) => (
              <label key={value} className="language-option">
                <input
                  type="checkbox"
                  checked={languages.includes(value)}
                  onChange={() => toggleLanguage(value)}
                />
                <span>{label}</span>
              </label>
            ))}
          </div>

          {error ? <p className="auth-form-error">{error}</p> : null}

          <div className="auth-submit-block">
            <button
              type="button"
              className="auth-primary-btn"
              onClick={handleSubmit}
              disabled={saving}
            >
              {saving ? "Guardando..." : "Enviar preferencias"}
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}