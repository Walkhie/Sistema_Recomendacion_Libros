import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recomendar_por_libro(codigo_semilla, dataframe, embeddings, top_n=10,
                         peso_texto=0.75, peso_ed=0.15, peso_citas=0.10):
    """
    """
    if codigo_semilla not in dataframe['Código del libro'].values:
        return {"error": "El código del libro no existe."}

    idx = dataframe[dataframe['Código del libro'] == codigo_semilla].index[0]

    # ── similitud sobre embeddings SBERT ──────────────────────
    similitud_texto = cosine_similarity(
        embeddings[idx].reshape(1, -1), embeddings
    ).flatten()
    # ─────────────────────────────────────────────────────────────────────────

    resultados = dataframe[['Código del libro', 'Titulo_Final', 'Autor_Final',
                             'W_Editorial_Norm', 'W_Citas_Norm']].copy()
    resultados['Similitud_Texto'] = similitud_texto

    # Ecuación maestra — intacta
    resultados['Score_Final'] = (
        (resultados['Similitud_Texto'] * peso_texto) +
        (resultados['W_Editorial_Norm'] * peso_ed) +
        (resultados['W_Citas_Norm'] * peso_citas)
    )

    # Excluir el libro semilla y retornar Top-N
    recomendaciones = (
        resultados[resultados['Código del libro'] != codigo_semilla]
        .sort_values('Score_Final', ascending=False)
        .head(top_n)
    )
    return recomendaciones.to_dict(orient='records')