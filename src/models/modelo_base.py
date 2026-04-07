import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Cargar la base de datos limpia (Ajusta la ruta si es necesario)
RUTA_DATOS = "data/preprocessing/Libros_Limpios_Recomendador.csv"
df = pd.read_csv(RUTA_DATOS)

# Asegurar que no haya nulos de última hora en las columnas clave
df['Tag'] = df['Tag'].fillna('')
df['W_Editorial_Norm'] = df['W_Editorial_Norm'].fillna(0.0)
df['W_Citas_Norm'] = df['W_Citas_Norm'].fillna(0.0)

# 2. Vectorizar el texto (Crear la matriz matemática)
# Usamos max_features para optimizar memoria si el catálogo es muy grande
print("Entrenando el modelo de lenguaje...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
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
    
    # 3. LA ECUACIÓN MAESTRA (NIVEL 1)
    resultados['Score_Final'] = (
        (resultados['Similitud_Texto'] * peso_texto) +
        (resultados['W_Editorial_Norm'] * peso_ed) +
        (resultados['W_Citas_Norm'] * peso_citas)
    )

    # NIVEL 1: Similitud estricta
    resultados_n1 = resultados[resultados['Similitud_Texto'] >= 0.17]
    recomendaciones = resultados_n1[resultados_n1['Código del libro'] != codigo_semilla].sort_values(by='Score_Final', ascending=False).head(top_n)
    
    # Lista de IDs recomendados para no repetir libros en los siguientes niveles
    libros_usados = set(recomendaciones['Código del libro'].tolist())
    libros_usados.add(codigo_semilla)

    # ==========================================
    # LÓGICA DE RELLENO PROGRESIVO (BACKFILLING)
    # ==========================================
    
    # Función auxiliar para añadir faltantes
    def agregar_faltantes(df_actual, df_nuevos):
        faltan = top_n - len(df_actual)
        if faltan <= 0 or df_nuevos.empty: return df_actual
        # Filtramos los que ya están en la lista
        df_nuevos = df_nuevos[~df_nuevos['Código del libro'].isin(libros_usados)]
        if df_nuevos.empty: return df_actual
        # Tomamos solo los necesarios y actualizamos la lista de usados
        agregados = df_nuevos.head(faltan)
        libros_usados.update(agregados['Código del libro'].tolist())
        return pd.concat([df_actual, agregados])

    # Extraemos info semilla
    kws_semilla = str(dataframe.loc[idx, 'Keywords']).lower()
    area_semilla = dataframe.loc[idx, 'Area_Conocimiento']
    
    # NIVEL 2: Keywords
    if len(recomendaciones) < top_n and kws_semilla != 'nan' and kws_semilla.strip():
        lista_kws = [k.strip() for k in kws_semilla.replace(';', ',').split(',') if len(k.strip()) > 3]
        
        def contar_kws(kw_texto):
            if pd.isna(kw_texto) or str(kw_texto).lower() == 'nan': return 0
            t = str(kw_texto).lower()
            return sum(1 for k in lista_kws if k in t)
            
        df_kws = dataframe.copy()
        df_kws['Kw_Match'] = df_kws['Keywords'].apply(contar_kws)
        df_kws = df_kws[df_kws['Kw_Match'] > 0].copy()
        
        if not df_kws.empty:
            df_kws['Score_Final'] = (df_kws['W_Editorial_Norm'] * 0.6) + (df_kws['W_Citas_Norm'] * 0.4)
            df_kws['Similitud_Texto'] = 0.0 # Flag visual de que es relleno
            df_kws = df_kws.sort_values(by=['Kw_Match', 'Score_Final'], ascending=[False, False])
            recomendaciones = agregar_faltantes(recomendaciones, df_kws)

    # NIVEL 3: Área de Conocimiento (Excluyendo 'General')
    if len(recomendaciones) < top_n and area_semilla != 'General':
        df_area = dataframe[dataframe['Area_Conocimiento'] == area_semilla].copy()
        df_area['Score_Final'] = (df_area['W_Editorial_Norm'] * 0.6) + (df_area['W_Citas_Norm'] * 0.4)
        df_area['Similitud_Texto'] = 0.0
        df_area = df_area.sort_values(by='Score_Final', ascending=False)
        recomendaciones = agregar_faltantes(recomendaciones, df_area)

    # NIVEL 4: Top Prestigio Global (El Salvavidas)
    if len(recomendaciones) < top_n:
        df_global = dataframe.copy()
        df_global['Score_Final'] = (df_global['W_Editorial_Norm'] * 0.6) + (df_global['W_Citas_Norm'] * 0.4)
        df_global['Similitud_Texto'] = 0.0
        df_global = df_global.sort_values(by='Score_Final', ascending=False)
        recomendaciones = agregar_faltantes(recomendaciones, df_global)

    columnas_retorno = ['Código del libro', 'Titulo_Final', 'Similitud_Texto', 'W_Editorial_Norm', 'W_Citas_Norm', 'Score_Final']
    return recomendaciones.head(top_n)[columnas_retorno]

# ==========================================
# ÁREA DE PRUEBAS
# ==========================================
if __name__ == "__main__":
    # Probamos con un libro específico (ajusta el código según tu base de datos)
    codigo_prueba = 'CEJ0114' 
    
    libro_info = df[df['Código del libro'] == codigo_prueba].iloc[0]
    print(f"\nLibro Semilla: {libro_info['Titulo_Final']} (Autor: {libro_info['Autor_Final']})")
    print("-" * 60)
    
    # Llamamos a la función
    top_10 = recomendar_por_libro(codigo_prueba, df, tfidf_matrix)
    
    # Mostramos los resultados en la terminal
    print(top_10[['Código del libro', 'Titulo_Final', 'Similitud_Texto', 'W_Editorial_Norm','W_Citas_Norm', 'Score_Final']])