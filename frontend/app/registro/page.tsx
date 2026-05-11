"use client";

import { FormEvent, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { FirebaseError } from "firebase/app";

import { registerOrContinueWithEmail } from "@/lib/auth";
import { createUserProfile, savePreferences } from "@/lib/userStore";

export default function RegistroPage() {
  const router = useRouter();

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleRegister = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");

    const cleanFirstName = firstName.trim();
    const cleanLastName = lastName.trim();
    const cleanEmail = email.trim().toLowerCase();

    if (!cleanFirstName || !cleanLastName) {
      setError("Completa nombre y apellidos.");
      return;
    }

    if (!cleanEmail) {
      setError("Completa el correo electrónico.");
      return;
    }

    if (password.length < 6) {
      setError("La contraseña debe tener al menos 6 caracteres.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Las contraseñas no coinciden.");
      return;
    }

    try {
      setLoading(true);

      const fullName = `${cleanFirstName} ${cleanLastName}`.trim();

      const { user } = await registerOrContinueWithEmail(
        cleanEmail,
        password,
        fullName
      );

      await createUserProfile(user.uid, {
        firstName: cleanFirstName,
        lastName: cleanLastName,
        email: user.email ?? cleanEmail,
      });

      await savePreferences(user.uid, {
        topics: [],
        favoriteSeedBookIds: [],
        onboardingCompleted: false,
      });

      router.replace("/onboarding/temas");
    } catch (err) {
      console.error(err);

      if (err instanceof FirebaseError) {
        if (err.code === "auth/email-already-in-use") {
          setError("Ese correo ya está registrado. Inicia sesión o usa otro correo.");
          return;
        }

        if (err.code === "auth/invalid-credential") {
          setError(
            "Ese correo ya existe y la contraseña no coincide. Inicia sesión o usa la contraseña correcta."
          );
          return;
        }

        if (err.code === "auth/weak-password") {
          setError("La contraseña debe tener al menos 6 caracteres.");
          return;
        }

        if (err.code === "auth/invalid-email") {
          setError("El correo electrónico no es válido.");
          return;
        }
      }

      setError("No fue posible crear la cuenta.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-flow-page">
      <section className="auth-flow-card">
        <Link href="/login" className="auth-flow-back" aria-label="Volver">
          ←
        </Link>

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
          <h1 className="auth-flow-section-title">Formulario de registro</h1>

          <form className="auth-form" onSubmit={handleRegister}>
            <div className="auth-form-grid-2">
              <div className="auth-form-field">
                <label htmlFor="firstName" className="auth-form-label">
                  Nombre
                </label>
                <input
                  id="firstName"
                  className="auth-form-input"
                  type="text"
                  value={firstName}
                  onChange={(event) => setFirstName(event.target.value)}
                  autoComplete="given-name"
                />
              </div>

              <div className="auth-form-field">
                <label htmlFor="lastName" className="auth-form-label">
                  Apellidos
                </label>
                <input
                  id="lastName"
                  className="auth-form-input"
                  type="text"
                  value={lastName}
                  onChange={(event) => setLastName(event.target.value)}
                  autoComplete="family-name"
                />
              </div>

              <div className="auth-form-field auth-form-field--full">
                <label htmlFor="email" className="auth-form-label">
                  Correo electrónico
                </label>
                <input
                  id="email"
                  className="auth-form-input"
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                />
              </div>

              <div className="auth-form-field auth-form-field--full">
                <label htmlFor="password" className="auth-form-label">
                  Contraseña
                </label>
                <input
                  id="password"
                  className="auth-form-input"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete="new-password"
                />
              </div>

              <div className="auth-form-field auth-form-field--full">
                <label htmlFor="confirmPassword" className="auth-form-label">
                  Confirmar contraseña
                </label>
                <input
                  id="confirmPassword"
                  className="auth-form-input"
                  type="password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  autoComplete="new-password"
                />
              </div>
            </div>

            {error ? <p className="auth-form-error">{error}</p> : null}

            <div className="auth-submit-block">
              <button
                type="submit"
                className="auth-primary-btn"
                disabled={loading}
              >
                {loading ? "Creando..." : "Registrarse"}
              </button>
            </div>
          </form>
        </div>
      </section>
    </main>
  );
}