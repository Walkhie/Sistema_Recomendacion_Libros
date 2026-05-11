"use client";

import { FormEvent, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";

import styles from "./page.module.css";
import { loginWithEmail, loginWithGoogle } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [error, setError] = useState("");
  const [loadingEmail, setLoadingEmail] = useState(false);
  const [loadingGoogle, setLoadingGoogle] = useState(false);

  const isLoading = loadingEmail || loadingGoogle;

  const handleLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
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
      setLoadingEmail(true);

      await loginWithEmail(email.trim().toLowerCase(), password);
      router.replace("/");
    } catch (err) {
      console.error(err);
      setError("No fue posible iniciar sesión. Verifica tus credenciales.");
    } finally {
      setLoadingEmail(false);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      setError("");
      setLoadingGoogle(true);

      await loginWithGoogle();
      router.replace("/");
    } catch (err) {
      console.error(err);
      setError("No fue posible iniciar sesión con Google.");
    } finally {
      setLoadingGoogle(false);
    }
  };

  return (
    <main className={styles.page}>
      <section className={styles.card}>
        <div className={styles.content}>
          <h1 className={styles.title}>
            Accede a todo el
            <br />
            conocimiento que
            <br />
            necesitas en un
            <br />
            mismo lugar
          </h1>

          <p className={styles.copy}>
            Encuentra recursos académicos relevantes para tus cursos e intereses.
            Descubre contenido confiable, recomendaciones personalizadas y explora
            autores e instituciones sin búsquedas interminables.
          </p>

          <form className={styles.form} onSubmit={handleLogin}>
            <div className={styles.field}>
              <label htmlFor="email" className={styles.label}>
                Usuario
              </label>
              <input
                id="email"
                className={styles.input}
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                autoComplete="email"
              />
            </div>

            <div className={styles.field}>
              <label htmlFor="password" className={styles.label}>
                Contraseña
              </label>
              <input
                id="password"
                className={styles.input}
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="current-password"
              />
            </div>

            {error ? <p className={styles.error}>{error}</p> : null}

            <div className={styles.row}>
              <button
                type="submit"
                disabled={isLoading}
                className={`${styles.buttonBase} ${styles.signInBtn}`}
              >
                {loadingEmail ? "Cargando..." : "Iniciar sesión"}
              </button>

              <Link
                href="/registro"
                className={`${styles.buttonBase} ${styles.registerBtn}`}
              >
                Registrarse
              </Link>
            </div>

            <button
              type="button"
              disabled={isLoading}
              className={`${styles.buttonBase} ${styles.googleBtn}`}
              onClick={handleGoogleLogin}
            >
              {loadingGoogle ? "Cargando..." : "Continuar con Google"}
            </button>
          </form>
        </div>

        <aside className={styles.rightPanel}>
          <div className={styles.logoWrap}>
            <Image
              src="/logoTitulo.png"
              alt="BookMatch"
              width={300}
              height={300}
              className={styles.logo}
              priority
            />
          </div>
        </aside>
      </section>
    </main>
  );
}