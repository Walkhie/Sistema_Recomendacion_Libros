from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CSV_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "preprocessing"
    / "Libros_Limpios_Recomendador.csv"
)


class RecommenderService:
    def __init__(self):
        print("Iniciando RecommenderService: Cargando datos y entrenando NLP...")

        if not CSV_PATH.exists():
            raise FileNotFoundError(f"No se encontró el archivo del catálogo en: {CSV_PATH}")

        self.df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

        # Limpieza rápida de nulos
        self.df["Tag"] = self.df["Tag"].fillna("")
        self.df["W_Editorial_Norm"] = self.df["W_Editorial_Norm"].fillna(0.0)
        self.df["W_Citas_Norm"] = self.df["W_Citas_Norm"].fillna(0.0)
        self.df["Keywords"] = self.df["Keywords"].fillna("")
        self.df["Area_Conocimiento"] = self.df["Area_Conocimiento"].fillna("General")
        self.df["Titulo_Final"] = self.df["Titulo_Final"].fillna("")
        self.df["Autor_Final"] = self.df["Autor_Final"].fillna("")

        # Normalizar tipo del id para evitar problemas de comparación
        self.df["Código del libro"] = self.df["Código del libro"].astype(str)

        # Vectorización: se hace una sola vez al arrancar la API
        self.vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["Tag"])
        print("✅ Modelo NLP cargado en memoria y listo para recomendar.")

    def recommend(
        self,
        book_id: str,
        top_n: int = 10,
        peso_texto: float = 0.75,
        peso_ed: float = 0.15,
        peso_citas: float = 0.10,
    ):
        book_id = str(book_id)

        if book_id not in self.df["Código del libro"].values:
            return {"error": "El código del libro no existe en el catálogo."}

        idx = self.df[self.df["Código del libro"] == book_id].index[0]

        similitud_texto = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()

        resultados = self.df[
            [
                "Código del libro",
                "Titulo_Final",
                "Autor_Final",
                "Area_Conocimiento",
                "W_Editorial_Norm",
                "W_Citas_Norm",
            ]
        ].copy()

        resultados["Similitud_Texto"] = similitud_texto
        resultados["Score_Final"] = (
            (resultados["Similitud_Texto"] * peso_texto)
            + (resultados["W_Editorial_Norm"] * peso_ed)
            + (resultados["W_Citas_Norm"] * peso_citas)
        )

        # Nivel 1: similitud textual fuerte
        resultados_n1 = resultados[resultados["Similitud_Texto"] >= 0.17].copy()
        resultados_n1["Nivel"] = "Nivel 1 (Texto)"

        recomendaciones = (
            resultados_n1[resultados_n1["Código del libro"] != book_id]
            .sort_values(by="Score_Final", ascending=False)
            .head(top_n)
        )

        libros_usados = set(recomendaciones["Código del libro"].astype(str).tolist())
        libros_usados.add(book_id)

        def agregar_faltantes(df_actual: pd.DataFrame, df_nuevos: pd.DataFrame) -> pd.DataFrame:
            faltan = top_n - len(df_actual)
            if faltan <= 0 or df_nuevos.empty:
                return df_actual

            df_nuevos = df_nuevos[
                ~df_nuevos["Código del libro"].astype(str).isin(libros_usados)
            ]

            if df_nuevos.empty:
                return df_actual

            agregados = df_nuevos.head(faltan).copy()
            libros_usados.update(agregados["Código del libro"].astype(str).tolist())

            return pd.concat([df_actual, agregados], ignore_index=True)

        kws_semilla = str(self.df.loc[idx, "Keywords"]).lower()
        area_semilla = str(self.df.loc[idx, "Area_Conocimiento"]).strip()

        # Nivel 2: keywords
        if len(recomendaciones) < top_n and kws_semilla.strip():
            lista_kws = [
                k.strip()
                for k in kws_semilla.replace(";", ",").split(",")
                if len(k.strip()) > 3
            ]

            def contar_kws(kw_texto: str) -> int:
                t = str(kw_texto).lower()
                return sum(1 for k in lista_kws if k in t)

            df_kws = self.df.copy()
            df_kws["Kw_Match"] = df_kws["Keywords"].apply(contar_kws)
            df_kws = df_kws[df_kws["Kw_Match"] > 0].copy()

            if not df_kws.empty:
                df_kws["Score_Final"] = (
                    (df_kws["W_Editorial_Norm"] * 0.6)
                    + (df_kws["W_Citas_Norm"] * 0.4)
                )
                df_kws["Similitud_Texto"] = 0.0
                df_kws["Nivel"] = "Nivel 2 (Keywords)"
                df_kws = df_kws.sort_values(
                    by=["Kw_Match", "Score_Final"],
                    ascending=[False, False],
                )

                recomendaciones = agregar_faltantes(
                    recomendaciones,
                    df_kws[
                        [
                            "Código del libro",
                            "Titulo_Final",
                            "Autor_Final",
                            "Area_Conocimiento",
                            "W_Editorial_Norm",
                            "W_Citas_Norm",
                            "Similitud_Texto",
                            "Score_Final",
                            "Nivel",
                        ]
                    ],
                )

        # Nivel 3: área
        if len(recomendaciones) < top_n and area_semilla and area_semilla != "General":
            df_area = self.df[self.df["Area_Conocimiento"] == area_semilla].copy()
            df_area["Score_Final"] = (
                (df_area["W_Editorial_Norm"] * 0.6)
                + (df_area["W_Citas_Norm"] * 0.4)
            )
            df_area["Similitud_Texto"] = 0.0
            df_area["Nivel"] = "Nivel 3 (Área)"
            df_area = df_area.sort_values(by="Score_Final", ascending=False)

            recomendaciones = agregar_faltantes(
                recomendaciones,
                df_area[
                    [
                        "Código del libro",
                        "Titulo_Final",
                        "Autor_Final",
                        "Area_Conocimiento",
                        "W_Editorial_Norm",
                        "W_Citas_Norm",
                        "Similitud_Texto",
                        "Score_Final",
                        "Nivel",
                    ]
                ],
            )

        # Nivel 4: salvavidas global
        if len(recomendaciones) < top_n:
            df_global = self.df.copy()
            df_global["Score_Final"] = (
                (df_global["W_Editorial_Norm"] * 0.6)
                + (df_global["W_Citas_Norm"] * 0.4)
            )
            df_global["Similitud_Texto"] = 0.0
            df_global["Nivel"] = "Nivel 4 (Salvavidas)"
            df_global = df_global.sort_values(by="Score_Final", ascending=False)

            recomendaciones = agregar_faltantes(
                recomendaciones,
                df_global[
                    [
                        "Código del libro",
                        "Titulo_Final",
                        "Autor_Final",
                        "Area_Conocimiento",
                        "W_Editorial_Norm",
                        "W_Citas_Norm",
                        "Similitud_Texto",
                        "Score_Final",
                        "Nivel",
                    ]
                ],
            )

        columnas_retorno = [
            "Código del libro",
            "Titulo_Final",
            "Autor_Final",
            "Area_Conocimiento",
            "Nivel",
            "Similitud_Texto",
            "W_Editorial_Norm",
            "W_Citas_Norm",
            "Score_Final",
        ]

        final_df = recomendaciones.head(top_n)[columnas_retorno].copy()
        final_df["Similitud_Texto"] = final_df["Similitud_Texto"].round(4)
        final_df["W_Editorial_Norm"] = final_df["W_Editorial_Norm"].round(4)
        final_df["W_Citas_Norm"] = final_df["W_Citas_Norm"].round(4)
        final_df["Score_Final"] = final_df["Score_Final"].round(4)

        return final_df.to_dict(orient="records")