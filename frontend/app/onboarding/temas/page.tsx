"use client";

import { useMemo, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { savePreferences } from "@/lib/userStore";
import { useAuth } from "@/context/AuthContext";

const TOPICS = [
  "arte",
  "historia",
  "cálculo",
  "programación",
  "química",
  "física",
  "música",
  "filosofía",
  "economía",
  "biología",
  "ciencias sociales",
  "medicina",
];

export default function TemasPage() {
  const router = useRouter();
  const { user, loading } = useAuth();

  const [topics, setTopics] = useState<string[]>([
    "cálculo",
    "programación",
    "química",
  ]);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const displayName = useMemo(() => {
    if (user?.displayName?.trim()) return user.displayName;
    if (user?.email) return user.email.split("@")[0];
    return "Nombre del Usuario";
  }, [user]);

  const toggleTopic = (topic: string) => {
    setTopics((prev) =>
      prev.includes(topic)
        ? prev.filter((item) => item !== topic)
        : [...prev, topic]
    );
  };

  const handleSubmit = async () => {
    if (!user) {
      setError("Debes iniciar sesión.");
      return;
    }

    if (topics.length === 0) {
      setError("Selecciona al menos un tema.");
      return;
    }

    try {
      setSaving(true);
      setError("");

      await savePreferences(user.uid, {
        topics,
        onboardingCompleted: true,
      });

      router.push("/");
    } catch (err) {
      console.error(err);
      setError("No se pudieron guardar los temas.");
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
          onClick={() => router.push("/onboarding/idiomas")}
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
            ¿Cuáles son tus temas de interés?
          </h1>

          <p className="auth-flow-section-copy">
            Esto nos ayuda a recomendarte libros que traten de tus temas favoritos.
            <strong> Podrás cambiar esta configuración en cualquier momento.</strong>
          </p>

          <div className="topic-grid">
            {TOPICS.map((topic) => {
              const active = topics.includes(topic);

              return (
                <button
                  key={topic}
                  type="button"
                  className={`topic-chip ${active ? "topic-chip--active" : ""}`}
                  onClick={() => toggleTopic(topic)}
                >
                  {topic}
                </button>
              );
            })}
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