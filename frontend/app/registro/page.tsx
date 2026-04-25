"use client";

import { FormEvent, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { registerWithEmail } from "@/lib/auth";
import { createUserProfile } from "@/lib/userStore";

export default function RegistroPage() {
  const router = useRouter();

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleRegister = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");

    if (!firstName.trim() || !lastName.trim()) {
      setError("Completa nombre y apellidos.");
      return;
    }

    if (!email.trim()) {
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

      const fullName = `${firstName} ${lastName}`.trim();
      const user = await registerWithEmail(email.trim(), password, fullName);

      await createUserProfile(user.uid, {
        firstName: firstName.trim(),
        lastName: lastName.trim(),
        email: user.email ?? email.trim(),
      });

      router.push("/onboarding/idiomas");
    } catch (err) {
      console.error(err);
      setError("No fue posible crear la cuenta.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-flow-page">
      <section className="auth-flow-card">
        <button
          type="button"
          className="auth-flow-back"
          onClick={() => router.push("/login")}
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
          <h1 className="auth-flow-section-title">Formulario De Registro</h1>

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
                  onChange={(e) => setFirstName(e.target.value)}
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
                  onChange={(e) => setLastName(e.target.value)}
                />
              </div>

              <div className="auth-form-field auth-form-field--full">
                <label htmlFor="email" className="auth-form-label">
                  Correo Electronico
                </label>
                <input
                  id="email"
                  className="auth-form-input"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
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
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>

              <div className="auth-form-field auth-form-field--full">
                <label htmlFor="confirmPassword" className="auth-form-label">
                  Contraseña
                </label>
                <input
                  id="confirmPassword"
                  className="auth-form-input"
                  type="password"
                  placeholder="Repita la contraseña"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </div>
            </div>

            {error ? <p className="auth-form-error">{error}</p> : null}

            <div className="auth-actions">
              <button type="submit" className="auth-primary-btn" disabled={loading}>
                {loading ? "Cargando..." : "Registrarse"}
              </button>
            </div>
          </form>
        </div>
      </section>
    </main>
  );
}