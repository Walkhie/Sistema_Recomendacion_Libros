"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";

import { savePreferences } from "@/lib/userStore";
import { useAuth } from "@/context/AuthContext";

const API_URL = "http://127.0.0.1:8000";

type TopicOption = {
  name: string;
  bookCount: number;
  totalCitations: number;
};

export default function TemasPage() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();

  const [availableTopics, setAvailableTopics] = useState<TopicOption[]>([]);
  const [topics, setTopics] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [loadingTopics, setLoadingTopics] = useState(true);

  const displayName = useMemo(() => {
    if (user?.displayName?.trim()) return user.displayName;
    if (user?.email) return user.email.split("@")[0];
    return "Nombre del Usuario";
  }, [user]);

  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    }
  }, [authLoading, user, router]);

  useEffect(() => {
    let active = true;

    async function loadTopics() {
      if (authLoading) return;

      if (!user) {
        setLoadingTopics(false);
        return;
      }

      try {
        setLoadingTopics(true);
        setError("");

        const response = await fetch(`${API_URL}/books/topics?limit=40`);

        if (!response.ok) {
          throw new Error("No se pudieron cargar los temas.");
        }

        const data: TopicOption[] = await response.json();

        if (!active) return;

        setAvailableTopics(data);
      } catch (err) {
        console.error(err);

        if (active) {
          setError("No se pudieron cargar los temas desde el catálogo.");
        }
      } finally {
        if (active) {
          setLoadingTopics(false);
        }
      }
    }

    loadTopics();

    return () => {
      active = false;
    };
  }, [authLoading, user]);

  const toggleTopic = (topic: string) => {
    setTopics((prev) =>
      prev.includes(topic)
        ? prev.filter((item) => item !== topic)
        : [...prev, topic]
    );
  };

  const handleSubmit = async () => {
    if (!user) {
      router.replace("/login");
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
        onboardingCompleted: false,
      });

      router.replace("/onboarding/libros");
    } catch (err) {
      console.error(err);
      setError("No se pudieron guardar los temas.");
    } finally {
      setSaving(false);
    }
  };

  if (authLoading || loadingTopics || !user) {
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

        <div className="auth-flow-content auth-flow-content--topics">
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
            Selecciona los temas que más se ajusten a tus intereses académicos.
            Estos temas vienen directamente del catálogo de libros.
            <strong>
              {" "}
              Luego te mostraremos los libros mejor posicionados por citaciones.
            </strong>
          </p>

          <div className="topic-grid">
            {availableTopics.map((topic) => {
              const active = topics.includes(topic.name);

              return (
                <button
                  key={topic.name}
                  type="button"
                  className={`topic-chip ${active ? "topic-chip--active" : ""}`}
                  onClick={() => toggleTopic(topic.name)}
                  title={`${topic.bookCount} libros · ${topic.totalCitations} citaciones`}
                >
                  {topic.name}
                </button>
              );
            })}
          </div>

          {availableTopics.length === 0 && !error ? (
            <p className="auth-form-error">
              No se encontraron temas disponibles en el catálogo.
            </p>
          ) : null}

          {error ? <p className="auth-form-error">{error}</p> : null}

          <div className="auth-submit-block">
            <button
              type="button"
              className="auth-primary-btn"
              onClick={handleSubmit}
              disabled={saving}
            >
              {saving ? "Guardando..." : "Continuar"}
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}