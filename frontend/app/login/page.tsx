"use client";

import { FormEvent, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { loginWithEmail } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");

    if (!email.trim()) {
      setError("Ingresa tu correo electrónico.");
      return;
    }

    if (!password.trim()) {
      setError("Ingresa tu contraseña.");
      return;
    }

    try {
      setLoading(true);
      await loginWithEmail(email.trim(), password);
      router.push("/");
    } catch (err) {
      console.error(err);
      setError("No fue posible iniciar sesión. Verifica tus credenciales.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-login-page">
      <section className="auth-login-card">
        <div className="auth-login-content">
          <div className="auth-login-logo">
            <Image
              src="/logoTitulo.png"
              alt="BookMatch"
              width={132}
              height={132}
              className="auth-login-logo-image"
              priority
            />
          </div>

          <h1 className="auth-login-title">
            Accede a todo el
            <br />
            conocimiento
            <br />
            que necesitas en un
            <br />
            mismo lugar
          </h1>

          <p className="auth-login-copy">
            Encuentra recursos académicos relevantes para tus cursos e intereses.
            Descubre contenido confiable, recomendaciones personalizadas y explora
            autores e instituciones sin búsquedas interminables.
          </p>

          <form className="auth-login-form" onSubmit={handleLogin}>
            <div className="auth-form-field">
              <label htmlFor="email" className="auth-form-label">
                Usuario
              </label>
              <input
                id="email"
                className="auth-form-input"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div className="auth-form-field" style={{ marginTop: 14 }}>
              <label htmlFor="password" className="auth-form-label">
                Contraseña
              </label>
              <input
                id="password"
                className="auth-form-input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            {error ? <p className="auth-form-error">{error}</p> : null}

            <div className="login-actions-row">
              <button
                type="submit"
                disabled={loading}
                className="login-outline-btn"
              >
                {loading ? "Cargando..." : "Iniciar sesion"}
              </button>

              <button
                type="button"
                className="auth-primary-btn login-register-btn"
                onClick={() => router.push("/registro")}
              >
                Registrarse
              </button>
            </div>
          </form>
        </div>
      </section>
    </main>
  );
}