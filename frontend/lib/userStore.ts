import {
  deleteDoc,
  doc,
  getDoc,
  serverTimestamp,
  setDoc,
} from "firebase/firestore";
import { db } from "./firebase";

type UserProfileInput = {
  firstName: string;
  lastName: string;
  email: string;
};

type PreferencesInput = {
  languages?: string[];
  topics?: string[];
  favoriteSeedBookIds?: string[];
  onboardingCompleted?: boolean;
};

type BookSummary = {
  id: string;
  title: string;
  authors?: string;
  category?: string;
  year?: string;
  citations?: number;
  editorialArea?: string;
};

export async function createUserProfile(uid: string, data: UserProfileInput) {
  await setDoc(
    doc(db, "users", uid),
    {
      firstName: data.firstName,
      lastName: data.lastName,
      fullName: `${data.firstName} ${data.lastName}`.trim(),
      email: data.email,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    },
    { merge: true }
  );
}

export async function savePreferences(uid: string, prefs: PreferencesInput) {
  await setDoc(
    doc(db, "users", uid, "preferences", "current"),
    {
      ...prefs,
      updatedAt: serverTimestamp(),
    },
    { merge: true }
  );
}

export async function saveFavorite(uid: string, book: BookSummary) {
  await setDoc(doc(db, "users", uid, "favorites", book.id), {
    bookId: book.id,
    title: book.title,
    authors: book.authors ?? "",
    category: book.category ?? "",
    year: book.year ?? "",
    citations: book.citations ?? 0,
    editorialArea: book.editorialArea ?? "",
    savedAt: serverTimestamp(),
  });
}

export async function removeFavorite(uid: string, bookId: string) {
  await deleteDoc(doc(db, "users", uid, "favorites", bookId));
}

export async function setRecommendationReaction(
  uid: string,
  bookId: string,
  reaction: "like" | "dislike"
) {
  await setDoc(
    doc(db, "users", uid, "recommendation_feedback", bookId),
    {
      bookId,
      reaction,
      source: "detail_modal",
      updatedAt: serverTimestamp(),
    },
    { merge: true }
  );
}

export async function clearRecommendationReaction(uid: string, bookId: string) {
  await deleteDoc(doc(db, "users", uid, "recommendation_feedback", bookId));
}

export async function getUserProfile(uid: string) {
  const snap = await getDoc(doc(db, "users", uid));
  return snap.exists() ? snap.data() : null;
}