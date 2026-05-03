# -*- coding: utf-8 -*-
"""
bert_model.py
Implementa el modelo de recomendación híbrido basado en similitud de texto (SBERT) y factores de prestigio (editorial y citas).

"""

import time

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np


# 1. Cargar la base de datos limpia (Ajusta la ruta si es necesario)
RUTA_DATOS = "data/preprocessing/Libros_Limpios_Recomendador.csv"
df = pd.read_csv(RUTA_DATOS)

# Asegurar que no haya nulos de última hora en las columnas clave
df['Tag'] = df['Tag'].fillna('')
df['W_Editorial_Norm'] = df['W_Editorial_Norm'].fillna(0.0)

# 2. Cargar el modelo SBERT preentrenado
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# Generar embeddings para los tags de los libros

embeddings = model.encode(df['Tag'].tolist(),
                          batch_size=64, 
                          show_progress_bar=True,
                          convert_to_numpy=True)


np.save('data/embeddings/embeddings.npy', embeddings)