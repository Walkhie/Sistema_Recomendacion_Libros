import {
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocs,
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

export type UserPreferences = PreferencesInput & {
  updatedAt?: unknown;
};

type BookSummary = {
  id: string;
  title: string;
  edition?: string;
  authors?: string;
  category?: string;
  year?: string;
  citations?: number;
  editorialCount?: number;
  editorialArea?: string;
  editorial?: string;
  doi?: string;
  abstract?: string;
  keywords?: string;
  language?: string;
  institution?: string;
};

export type FavoriteBook = BookSummary & {
  bookId: string;
  savedAt?: unknown;
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

export async function getPreferences(
  uid: string
): Promise<UserPreferences | null> {
  const snap = await getDoc(doc(db, "users", uid, "preferences", "current"));
  return snap.exists() ? (snap.data() as UserPreferences) : null;
}

export async function saveFavorite(uid: string, book: BookSummary) {
  await setDoc(doc(db, "users", uid, "favorites", book.id), {
    bookId: book.id,
    title: book.title,
    edition: book.edition ?? "",
    authors: book.authors ?? "",
    category: book.category ?? "",
    year: book.year ?? book.edition ?? "",
    citations: book.citations ?? 0,
    editorialCount: book.editorialCount ?? 0,
    editorialArea: book.editorialArea ?? "",
    editorial: book.editorial ?? "",
    doi: book.doi ?? "",
    abstract: book.abstract ?? "",
    keywords: book.keywords ?? "",
    language: book.language ?? "",
    institution: book.institution ?? "",
    savedAt: serverTimestamp(),
  });
}

export async function removeFavorite(uid: string, bookId: string) {
  await deleteDoc(doc(db, "users", uid, "favorites", bookId));
}

export async function getUserFavorites(uid: string): Promise<FavoriteBook[]> {
  const snap = await getDocs(collection(db, "users", uid, "favorites"));

  return snap.docs.map((favoriteDoc) => {
    const data = favoriteDoc.data() as Partial<FavoriteBook>;

    return {
      id: favoriteDoc.id,
      bookId: data.bookId ?? favoriteDoc.id,
      title: data.title ?? "Libro sin título",
      edition: data.edition ?? data.year ?? "",
      authors: data.authors ?? "",
      category: data.category ?? "",
      year: data.year ?? data.edition ?? "",
      citations: data.citations ?? 0,
      editorialCount: data.editorialCount ?? 0,
      editorialArea: data.editorialArea ?? "",
      editorial: data.editorial ?? "",
      doi: data.doi ?? "",
      abstract: data.abstract ?? "",
      keywords: data.keywords ?? "",
      language: data.language ?? "",
      institution: data.institution ?? "",
      savedAt: data.savedAt,
    };
  });
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