import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import unicodedata
 
from recomendator import recomendar_por_libro 
 
# ── Configuración ─────────────────────────────────────────────────────────────
 
RUTA_CSV        = "data/preprocessing/Libros_Limpios_Recomendador.csv"
RUTA_EMBEDDINGS = "data/embeddings/embeddings.npy"

N_SEMILLAS  = 3
TOP_N       = 5
PESO_TEXTO  = 0.75
PESO_ED     = 0.15
PESO_CITAS  = 0.10
SEMILLA_RNG = None


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_datos(ruta_csv, ruta_embeddings):
    df = pd.read_csv(ruta_csv)
    embeddings = np.load(ruta_embeddings)

    if len(df) != len(embeddings):
        raise ValueError("Mismatch entre número de libros y embeddings")

    return df, embeddings


# ── Impresión de resultados ───────────────────────────────────────────────────
 
MEDALLAS = {1: "🥇", 2: "🥈", 3: "🥉"}
SEP      = "═" * 72
 
def imprimir_resultado(semilla, recomendaciones):
 
    print(f"\n{SEP}")
    print(f"    LIBRO SEMILLA")
    print(f"{'─' * 72}")
    print(f"  Código : {semilla['Código del libro']}")
    print(f"  Título : {semilla['Titulo_Final']}")
    print(f"  Autor  : {semilla['Autor_Final']}")
    print(f"{'─' * 72}")
    print(f"\n   TOP {TOP_N} RECOMENDACIONES\n")
    print(f"  {'Pos':<5} {'Código':<10} {'Score':>6}   {'Título'}")
    print(f"  {'─' * 68}")
 
    for i, libro in enumerate(recomendaciones, start=1):
        medalla = MEDALLAS.get(i, f"  {i}.")
        titulo  = libro['Titulo_Final']
        titulo  = (titulo[:52] + "…") if len(titulo) > 52 else titulo
        autor   = str(libro['Autor_Final'])
        autor   = (autor[:28] + "…") if len(autor) > 28 else autor
 
        print(f"  {medalla:<6} {libro['Código del libro']:<10} "
              f"{libro['Score_Final']:>6.4f}   {titulo}")
        print(f"  {'':6} {'':10}          ↳ {autor}")
        print(f"  {'':6} Sim.Texto={libro['Similitud_Texto']:.4f}  "
              f"W_Editorial={libro['W_Editorial_Norm']:.4f}  "
              f"W_Citas={libro['W_Citas_Norm']:.4f}")
        print()
 
    print(f"   Score = {PESO_TEXTO}·Sim.Texto "
          f"+ {PESO_ED}·W_Editorial + {PESO_CITAS}·W_Citas")
    print(SEP)
 
 # Normalizador 

def normalizar(texto):
    texto = texto.lower().strip()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

# Pruebas libros

def evaluar_libros(libros_evaluacion, df, embeddings, top_n=5):

    resultados_totales = []

    for titulo in libros_evaluacion:

        titulo_norm = normalizar(titulo)

        match = df[df['titulo_norm'] == titulo_norm]

        if match.empty:
            print(f"No encontrado: {titulo}")
            continue

        codigo = match.iloc[0]['Código del libro']

        recomendaciones = recomendar_por_libro(
            codigo_semilla=codigo,
            dataframe=df,
            embeddings=embeddings,
            top_n=top_n
        )

        if isinstance(recomendaciones, dict):
            continue

        # Guardar resultados en formato tabular
        for rank, rec in enumerate(recomendaciones, start=1):
            resultados_totales.append({
                "Titulo_Semilla": titulo,
                "Codigo_Semilla": codigo,
                "Rank": rank,
                "Codigo_Recomendado": rec['Código del libro'],
                "Titulo_Recomendado": rec['Titulo_Final'],
                "Score": rec['Score_Final']
            })

    return pd.DataFrame(resultados_totales)
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():

    df, embeddings = cargar_datos(RUTA_CSV, RUTA_EMBEDDINGS)

    rng = np.random.default_rng(SEMILLA_RNG)

    codigos = df["Código del libro"].sample(
        n=N_SEMILLAS,
        random_state=int(rng.integers(1000))
    ).tolist()

    print(f"\n{SEP}")
    print("   VALIDACIÓN — recomendar_por_libro()")
    print(SEP)
    print(f"\n   Semillas : {codigos}")

    for codigo in codigos:

        semilla = df[df["Código del libro"] == codigo].iloc[0]

        resultado = recomendar_por_libro(
            codigo_semilla=codigo,
            dataframe=df,
            embeddings=embeddings,
            top_n=TOP_N,
            peso_texto=PESO_TEXTO,
            peso_ed=PESO_ED,
            peso_citas=PESO_CITAS,
        )

        if isinstance(resultado, dict) and "error" in resultado:
            print(f"\n   {resultado['error']}")
            continue

        imprimir_resultado(semilla, resultado)

    # Validacion con modelo anterior

    libros_evaluacion = ["Salud mental y desplazamiento forzado","ESTADISTICA DESCRIPTIVA Y PROBABILIDAD", 
                         "La pandemia de la Covid-19 y sus efectos colaterales","¿Qué es la bioética?",
                         "LA DEMOCRACIA COMO FORMA DE VIDA","Reflexiones en torno a derechos humanos y grupos vulnerables",
                         "Prácticas de laboratorio en química general","Geopolítica, una forma de mirar al mundo",
                         "PAZ RECONCILIACION Y JUSTICIA TRANSICIONAL EN COLOMBIA Y AMERICA LATINA","ESTUDIOS SOBRE EL ARTE Y LA ARQUITECTURA COLONIALES EN COLOMBIA",
                         "Desafíos migratorios: realidades desde diversas orillas","SALUD PUBLICA EN COLOMBIA",
                         "Masculinidad como arma del silencio","HOMBRES TRANS Y LIBRETA MILITAR EN COLOMBIA",
                         "La fotografía, un documento social","DISEÑO Y EDUCACION","DIDACTICAS EN LA UNIVERSIDAD PERSPECTIVAS DESDE LA DOCENCIA",
                         "INVESTIGAR EN EDUCACION","MUSICA URBANA JUVENTUD Y RESISTENCIA UN VIAJE POR ALGUNOS SONIDOS UNDERGROUND DE AMERICA LATINA",
                         "Fauna y flora de la Unión, Antioquia"
                         ]
    
    df['titulo_norm'] = df['Titulo_Final'].apply(normalizar)
    
    df_resultados = evaluar_libros(libros_evaluacion, df, embeddings, top_n=5)

    df_resultados.to_excel("resultados_recomendacion.xlsx", index=False)
    

if __name__ == "__main__":
    main()