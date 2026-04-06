import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Cargar la base de datos limpia (Ajusta la ruta si es necesario)
RUTA_DATOS = "data/Libros_Limpios_Recomendador.csv"
df = pd.read_csv(RUTA_DATOS)

# Asegurar que no haya nulos de última hora en las columnas clave
df['Tag'] = df['Tag'].fillna('')
df['W_Editorial_Norm'] = df['W_Editorial_Norm'].fillna(0.0)
df['W_Citas_Norm'] = df['W_Citas_Norm'].fillna(0.0)

# 2. Vectorizar el texto (Crear la matriz matemática)
# Usamos max_features para optimizar memoria si el catálogo es muy grande
print("Entrenando el modelo de lenguaje...")
vectorizer = TfidfVectorizer(max_features=10000)
tfidf_matrix = vectorizer.fit_transform(df['Tag'])

def recomendar_por_libro(codigo_semilla, dataframe, matriz_tfidf, top_n=10, 
                         peso_texto=0.75, peso_ed=0.15, peso_citas=0.10):
    """
    Genera recomendaciones híbridas basadas en un libro semilla.
    """
    # Verificar que el libro exista en la base de datos
    if codigo_semilla not in dataframe['Código del libro'].values:
        return "Error: El código del libro no existe."
        
    # Obtener el índice (la fila) del libro semilla
    idx = dataframe[dataframe['Código del libro'] == codigo_semilla].index[0]
    
    # Calcular la Similitud del Coseno (Texto) de este libro contra TODOS los demás
    similitud_texto = cosine_similarity(matriz_tfidf[idx], matriz_tfidf).flatten()
    
    # Crear un DataFrame temporal solo con lo que necesitamos ver
    resultados = dataframe[['Código del libro', 'Titulo_Final', 'Autor_Final', 
                            'W_Editorial_Norm', 'W_Citas_Norm']].copy()
    
    # Agregar el puntaje de similitud de texto a los resultados
    resultados['Similitud_Texto'] = similitud_texto
    
    # 3. LA ECUACIÓN MAESTRA (Modelo Híbrido)
    resultados['Score_Final'] = (
        (resultados['Similitud_Texto'] * peso_texto) +
        (resultados['W_Editorial_Norm'] * peso_ed) +
        (resultados['W_Citas_Norm'] * peso_citas)
    )

    # Filtrar por un umbral mínimo de similitud de texto para asegurar priorización de relevancia
    resultados = resultados[resultados['Similitud_Texto'] >= 0.15]

    # Excluir el libro semilla para que no se recomiende a sí mismo
    recomendaciones = resultados[resultados['Código del libro'] != codigo_semilla]
    
    # Ordenar de mayor a menor basándonos en el Score Final
    recomendaciones = recomendaciones.sort_values(by='Score_Final', ascending=False)
    
    # Devolver los mejores resultados
    return recomendaciones.head(top_n)

# ==========================================
# ÁREA DE PRUEBAS
# ==========================================
if __name__ == "__main__":
    # Probamos con un libro específico (ajusta el código según tu base de datos)
    codigo_prueba = 'UGU0056' 
    
    libro_info = df[df['Código del libro'] == codigo_prueba].iloc[0]
    print(f"\nLibro Semilla: {libro_info['Titulo_Final']} (Autor: {libro_info['Autor_Final']})")
    print("-" * 60)
    
    # Llamamos a la función
    top_10 = recomendar_por_libro(codigo_prueba, df, tfidf_matrix)
    
    # Mostramos los resultados en la terminal
    print(top_10[['Código del libro', 'Titulo_Final', 'Similitud_Texto', 'W_Editorial_Norm','W_Citas_Norm', 'Score_Final']])