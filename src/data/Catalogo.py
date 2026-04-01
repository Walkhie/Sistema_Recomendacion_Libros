# -*- coding: utf-8 -*-
"""
catalogos_unificados.py
=======================
Enriquece un catálogo de libros (Libros.xlsx) consultando la API de OpenAlex
y produce dos archivos de salida:
  - Libros_Unificados_Recomendador.csv
  - Libros_Unificados_Recomendador.xlsx

Estrategia de búsqueda (en orden de prioridad):
  Fase 1 — Lote de DOIs   → coincidencia exacta, similitud = 1.0
  Fase 2 — Título + Autor → búsqueda textual en paralelo, umbral ≥ 0.50
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
EMAIL_CONTACTO    = "juan.riosp@upb.edu.co"
ARCHIVO_ENTRADA   = r"data\raw\Libros.xlsx" # Ruta al archivo de entrada con el catálogo original
SALIDA_CSV        = "Libros_Unificados_Recomendador.csv"
SALIDA_XLSX       = "Libros_Unificados_Recomendador.xlsx"

BASE_URL_OA       = "https://api.openalex.org/works"
TAMANO_LOTE_DOI   = 50
MAX_WORKERS       = 5          # hilos para la Fase 2
UMBRAL_SIMILITUD  = 0.50
MAX_REINTENTOS    = 3

DOI_REGEX = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
AUTORES_INVALIDOS = {"sin datos", "varios autores", "nan", ""}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# FUNCIONES DE LIMPIEZA
# =============================================================================

def extraer_doi(valor) -> str | None:
    """Extrae y normaliza el DOI de una celda que puede tener distintos formatos."""
    if pd.isna(valor):
        return None
    s = str(valor).strip().replace("doi:", "").strip()
    m = DOI_REGEX.search(s)
    if not m:
        return None
    return m.group(1).lower().strip().strip(".")


def limpiar_titulo(titulo) -> str:
    """Elimina paréntesis y mueve artículos pospuestos al inicio."""
    if pd.isna(titulo):
        return ""
    t = re.sub(r"\(.*?\)", "", str(titulo)).strip()
    t = re.sub(
        r"(.*),\s*(LA|EL|LOS|LAS|UN|UNA|UNOS|UNAS)$",
        r"\2 \1",
        t,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", t).strip()


def limpiar_autor(autor_raw) -> str:
    """Toma solo el primer autor de una lista separada por ';'."""
    if pd.isna(autor_raw):
        return ""
    return str(autor_raw).split(";")[0].strip()

# =============================================================================
# FUNCIONES DE SIMILITUD
# =============================================================================

def similitud_texto(a, b) -> float:
    """Ratio de similitud entre dos cadenas (case-insensitive)."""
    if not a or not b or pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def similitud_compuesta(tit_local, tit_oa, aut_local, aut_oa) -> float:
    """
    Combina similitud de título (70 %) y autor (30 %).
    Si el autor local no es válido, el score depende 100 % del título.
    """
    sim_tit = similitud_texto(tit_local, tit_oa)
    if str(aut_local).lower().strip() in AUTORES_INVALIDOS:
        return sim_tit
    sim_aut = similitud_texto(aut_local, aut_oa)
    return sim_tit * 0.70 + sim_aut * 0.30

# =============================================================================
# EXTRACCIÓN DE DATOS DE OPENALEX
# =============================================================================

def extraer_datos_oa(work: dict) -> dict:
    """Extrae los campos relevantes de un objeto 'work' de la API de OpenAlex."""
    conceptos = [
        c["display_name"]
        for c in work.get("concepts", [])
        if c.get("level", 3) <= 2 and c.get("score", 0) > 0.4
    ]
    autores = work.get("authorships", [])
    primer_autor = autores[0]["author"]["display_name"] if autores else ""

    return {
        "id":       work.get("id", ""),
        "title":    work.get("title") or work.get("display_name", ""),
        "author":   primer_autor,
        "concepts": "; ".join(conceptos),
        "citations": work.get("cited_by_count", 0),
    }

# =============================================================================
# FASE 1 — BÚSQUEDA POR LOTE DE DOIs
# =============================================================================

def fase1_busqueda_doi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consulta OpenAlex enviando hasta TAMANO_LOTE_DOI DOIs por petición.
    Actualiza las filas del DataFrame cuyo DOI coincide.
    """
    log.info("Fase 1: Búsqueda por lotes de DOI...")

    dois_unicos = df.loc[df["DOI_Limpio"].notna(), "DOI_Limpio"].unique()

    for i in tqdm(range(0, len(dois_unicos), TAMANO_LOTE_DOI), desc="Lotes DOI"):
        lote = dois_unicos[i : i + TAMANO_LOTE_DOI]
        params = {
            "filter":   "doi:" + "|".join(lote),
            "per_page": 200,
            "mailto":   EMAIL_CONTACTO,
        }

        try:
            r = requests.get(BASE_URL_OA, params=params, timeout=20)
            if r.status_code != 200:
                log.warning("Error en lote DOI (HTTP %s)", r.status_code)
                continue

            for work in r.json().get("results", []):
                doi_devuelto = (work.get("doi") or "").replace("https://doi.org/", "").strip()
                if not doi_devuelto:
                    continue

                datos = extraer_datos_oa(work)
                mascara = df["DOI_Limpio"] == doi_devuelto
                df.loc[mascara, "OpenAlex_ID"]       = datos["id"]
                df.loc[mascara, "OpenAlex_Concepts"]  = datos["concepts"]
                df.loc[mascara, "OpenAlex_Citations"] = datos["citations"]
                df.loc[mascara, "OpenAlex_Similarity"] = 1.0
                df.loc[mascara, "Metodo_Match"]       = "DOI (Lote)"

        except requests.RequestException as e:
            log.error("Error de conexión en lote DOI: %s", e)

        time.sleep(0.5)

    return df

# =============================================================================
# FASE 2 — FALLBACK POR TÍTULO + AUTOR (MULTIHILO)
# =============================================================================

def _procesar_libro(args: tuple) -> tuple:
    """
    Función ejecutada por cada hilo en la Fase 2.
    Intenta dos búsquedas textuales en OpenAlex:
      Plan B — título + autor
      Plan C — solo título
    """
    idx, fila = args
    titulo = str(fila["Titulo_Limpio"]).strip()
    autor  = str(fila["Autor_Limpio"]).strip()

    vacio = {
        "OpenAlex_ID": None, "OpenAlex_Title": None, "OpenAlex_Author": None,
        "OpenAlex_Concepts": None, "OpenAlex_Citations": 0,
        "Metodo_Match": "No Encontrado", "OpenAlex_Similarity": 0.0,
    }

    if not titulo or titulo == "nan":
        return idx, vacio

    planes = [
        ({"search": f"{titulo} {autor}".strip(), "mailto": EMAIL_CONTACTO}, "Titulo+Autor"),
        ({"search": titulo,                       "mailto": EMAIL_CONTACTO}, "Solo Titulo"),
    ]

    for intento in range(MAX_REINTENTOS):
        for params, metodo in planes:
            try:
                r = requests.get(BASE_URL_OA, params=params, timeout=10)

                if r.status_code == 429:
                    time.sleep(5)
                    break  # reintentar el ciclo externo

                if r.status_code != 200:
                    continue

                data = r.json()
                if not data.get("meta", {}).get("count", 0):
                    continue

                work  = data["results"][0]
                datos = extraer_datos_oa(work)
                sim   = similitud_compuesta(titulo, datos["title"], autor, datos["author"])

                if sim >= UMBRAL_SIMILITUD:
                    return idx, {
                        "OpenAlex_ID":       datos["id"],
                        "OpenAlex_Title":    datos["title"],
                        "OpenAlex_Author":   datos["author"],
                        "OpenAlex_Concepts": datos["concepts"],
                        "OpenAlex_Citations": datos["citations"],
                        "Metodo_Match":      metodo,
                        "OpenAlex_Similarity": sim,
                    }

            except requests.RequestException:
                time.sleep(3)

        else:
            # Ambos planes terminaron sin éxito en este intento
            return idx, vacio

    return idx, vacio


def fase2_fallback_titulo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para los libros sin coincidencia, lanza búsquedas en paralelo por título.
    """
    faltantes = df[df["Metodo_Match"] == "No Encontrado"]
    log.info("Fase 2: Fallback por Título (%d libros sin match)...", len(faltantes))

    resultados: dict = {}
    tareas_input = list(faltantes.iterrows())

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros = {executor.submit(_procesar_libro, args): args[0] for args in tareas_input}
        for futuro in tqdm(as_completed(futuros), total=len(futuros), desc="Plan B/C"):
            idx, res = futuro.result()
            resultados[idx] = res
            time.sleep(0.1)

    for idx, res in resultados.items():
        for col, val in res.items():
            df.at[idx, col] = val

    return df

# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos(ruta: str) -> pd.DataFrame:
    """Lee el Excel y devuelve el DataFrame con columnas renombradas y limpias."""
    df = pd.read_excel(ruta)

    col_doi = next(c for c in df.columns if "doi" in c.lower())
    df = df.rename(columns={
        "Libro":                                              "Titulo",
        "Autor Principal":                                    "Autor",
        col_doi:                                              "DOI_Original",
        "Resumen":                                            "Abstract",
        "Palabras Clave (separadas por \";\")":              "Keywords",
        "Idioma de publicación de la obra":                   "Idioma",
    })

    columnas = [
        "Código del libro", "Titulo", "Autor", "Año de publicación", "DOI_Original",
        "Editorial", "Abstract", "Keywords", "BISAC Catálogo", "Thema Catálogo",
        "Clasificación Dewey",
        "Institución coeditora (separar cada institución con \";\")",
        "Idioma",
    ]
    return df[[c for c in columnas if c in df.columns]].copy()


def preparar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas auxiliares de búsqueda y resultados."""
    df["DOI_Limpio"]          = df["DOI_Original"].apply(extraer_doi)
    df["Titulo_Limpio"]       = df["Titulo"].apply(limpiar_titulo)
    df["Autor_Limpio"]        = df["Autor"].apply(limpiar_autor)

    df["OpenAlex_ID"]         = None
    df["OpenAlex_Title"]      = None
    df["OpenAlex_Author"]     = None
    df["OpenAlex_Concepts"]   = None
    df["OpenAlex_Citations"]  = 0
    df["OpenAlex_Similarity"] = 0.0
    df["Metodo_Match"]        = "No Encontrado"
    return df

# =============================================================================
# CÁLCULO DE MÉTRICAS FINALES
# =============================================================================

def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de área de conocimiento y pesos editoriales."""
    df["Area_Conocimiento"] = (
        df["BISAC Catálogo"]
        .astype(str)
        .str.split(">").str[0]
        .str.replace(r"\[.*?\]", "", regex=True)
        .str.strip()
        .replace("nan", "General")
    )

    prestigio = (
        df.groupby(["Editorial", "Area_Conocimiento"])
        .size()
        .reset_index(name="Titulos_Editorial_Area")
    )
    df = df.merge(prestigio, on=["Editorial", "Area_Conocimiento"], how="left")

    df["W_Editorial"] = 1 + (0.1 * np.log1p(df["Titulos_Editorial_Area"]))
    df["W_Citas"]     = 0.5 * np.log1p(df["OpenAlex_Citations"])
    return df

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main() -> None:
    log.info("Cargando datos desde '%s'...", ARCHIVO_ENTRADA)
    df = cargar_datos(ARCHIVO_ENTRADA)
    df = preparar_columnas(df)

    df = fase1_busqueda_doi(df)
    df = fase2_fallback_titulo(df)
    df = calcular_metricas(df)

    # Eliminar columnas auxiliares de trabajo
    df = df.drop(columns=["DOI_Limpio", "Titulo_Limpio", "Autor_Limpio"], errors="ignore")

    log.info("\n--- Resumen de coincidencias ---\n%s", df["Metodo_Match"].value_counts().to_string())

    df.to_csv(SALIDA_CSV,  index=False, encoding="utf-8-sig")
    df.to_excel(SALIDA_XLSX, index=False)
    log.info("Archivos guardados: '%s' y '%s'", SALIDA_CSV, SALIDA_XLSX)


if __name__ == "__main__":
    main()