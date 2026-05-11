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

export type BookSummary = {
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

export type RecommendationReaction = "like" | "dislike";

export type RecommendationFeedback = {
  id: string;
  bookId: string;
  sourceBookId: string;
  reaction: RecommendationReaction;
  recommendedBook: BookSummary;
  sourceBook: BookSummary;
  updatedAt?: unknown;
};

type RecommendationReactionInput = {
  book: BookSummary;
  sourceBook: BookSummary;
  reaction: RecommendationReaction;
};

function normalizeBookSummary(
  book: Partial<BookSummary>,
  fallback?: Partial<BookSummary>
): BookSummary {
  return {
    id: book.id ?? fallback?.id ?? "",
    title: book.title ?? fallback?.title ?? "Libro sin título",
    edition: book.edition ?? fallback?.edition ?? "",
    authors: book.authors ?? fallback?.authors ?? "",
    category: book.category ?? fallback?.category ?? "",
    year: book.year ?? fallback?.year ?? book.edition ?? fallback?.edition ?? "",
    citations: book.citations ?? fallback?.citations ?? 0,
    editorialCount: book.editorialCount ?? fallback?.editorialCount ?? 0,
    editorialArea: book.editorialArea ?? fallback?.editorialArea ?? "",
    editorial: book.editorial ?? fallback?.editorial ?? "",
    doi: book.doi ?? fallback?.doi ?? "",
    abstract: book.abstract ?? fallback?.abstract ?? "",
    keywords: book.keywords ?? fallback?.keywords ?? "",
    language: book.language ?? fallback?.language ?? "",
    institution: book.institution ?? fallback?.institution ?? "",
  };
}

function makeRecommendationFeedbackId(sourceBookId: string, bookId: string) {
  return `${encodeURIComponent(sourceBookId)}__${encodeURIComponent(bookId)}`;
}

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
  const normalizedBook = normalizeBookSummary(book);

  await setDoc(doc(db, "users", uid, "favorites", normalizedBook.id), {
    ...normalizedBook,
    bookId: normalizedBook.id,
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
    const normalizedBook = normalizeBookSummary(data, {
      id: data.bookId ?? favoriteDoc.id,
    });

    return {
      ...normalizedBook,
      id: favoriteDoc.id,
      bookId: data.bookId ?? favoriteDoc.id,
      savedAt: data.savedAt,
    };
  });
}

export async function setRecommendationReaction(
  uid: string,
  input: RecommendationReactionInput
) {
  const recommendedBook = normalizeBookSummary(input.book);
  const sourceBook = normalizeBookSummary(input.sourceBook);

  if (!recommendedBook.id || !sourceBook.id) {
    throw new Error("No se pudo guardar la reacción: falta el libro recomendado o el libro origen.");
  }

  await setDoc(
    doc(
      db,
      "users",
      uid,
      "recommendation_feedback",
      makeRecommendationFeedbackId(sourceBook.id, recommendedBook.id)
    ),
    {
      bookId: recommendedBook.id,
      sourceBookId: sourceBook.id,
      reaction: input.reaction,
      recommendedBook,
      sourceBook,
      source: "detail_modal",
      updatedAt: serverTimestamp(),
    },
    { merge: true }
  );
}

export async function clearRecommendationReaction(
  uid: string,
  sourceBookId: string,
  bookId: string
) {
  await deleteDoc(
    doc(
      db,
      "users",
      uid,
      "recommendation_feedback",
      makeRecommendationFeedbackId(sourceBookId, bookId)
    )
  );
}

export async function getUserRecommendationFeedback(
  uid: string
): Promise<RecommendationFeedback[]> {
  const snap = await getDocs(
    collection(db, "users", uid, "recommendation_feedback")
  );

  return snap.docs.map((feedbackDoc) => {
    const data = feedbackDoc.data() as {
      bookId?: string;
      sourceBookId?: string;
      reaction?: RecommendationReaction;
      recommendedBook?: Partial<BookSummary>;
      book?: Partial<BookSummary>;
      sourceBook?: Partial<BookSummary>;
      updatedAt?: unknown;
    };

    const bookId =
      data.bookId ??
      data.recommendedBook?.id ??
      data.book?.id ??
      "";

    const sourceBookId =
      data.sourceBookId ??
      data.sourceBook?.id ??
      "";

    return {
      id: feedbackDoc.id,
      bookId,
      sourceBookId,
      reaction: data.reaction === "dislike" ? "dislike" : "like",
      recommendedBook: normalizeBookSummary(
        data.recommendedBook ?? data.book ?? {},
        { id: bookId }
      ),
      sourceBook: normalizeBookSummary(data.sourceBook ?? {}, {
        id: sourceBookId,
        title: "Libro base no disponible",
      }),
      updatedAt: data.updatedAt,
    };
  });
}

export async function getUserProfile(uid: string) {
  const snap = await getDoc(doc(db, "users", uid));
  return snap.exists() ? snap.data() : null;
}