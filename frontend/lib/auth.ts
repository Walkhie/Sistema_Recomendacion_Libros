import { FirebaseError } from "firebase/app";
import {
  GoogleAuthProvider,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  updateProfile,
} from "firebase/auth";
import { auth } from "./firebase";

const googleProvider = new GoogleAuthProvider();

export async function registerWithEmail(
  email: string,
  password: string,
  fullName: string
) {
  const cred = await createUserWithEmailAndPassword(auth, email, password);

  await updateProfile(cred.user, {
    displayName: fullName,
  });

  return cred.user;
}

export async function registerOrContinueWithEmail(
  email: string,
  password: string,
  fullName: string
) {
  try {
    const user = await registerWithEmail(email, password, fullName);
    return { user, mode: "created" as const };
  } catch (error) {
    if (
      error instanceof FirebaseError &&
      error.code === "auth/email-already-in-use"
    ) {
      const cred = await signInWithEmailAndPassword(auth, email, password);

      if (!cred.user.displayName?.trim() && fullName.trim()) {
        await updateProfile(cred.user, {
          displayName: fullName,
        });
      }

      return { user: cred.user, mode: "existing" as const };
    }

    throw error;
  }
}

export async function loginWithEmail(email: string, password: string) {
  const cred = await signInWithEmailAndPassword(auth, email, password);
  return cred.user;
}

export async function loginWithGoogle() {
  const cred = await signInWithPopup(auth, googleProvider);
  return cred.user;
}

export async function logoutUser() {
  await signOut(auth);
}