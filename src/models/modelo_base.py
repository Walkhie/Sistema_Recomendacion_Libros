# -*- coding: utf-8 -*-
"""
modelo_base.py
Implementa el modelo de recomendación híbrido basado en similitud de texto (TF-IDF) y factores de prestigio (editorial y citas).
Incluye una lógica de relleno progresivo (backfilling) para garantizar que siempre se devuelvan 10 recomendaciones.

"""

import time
from tabulate import tabulate
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
    Genera recomendaciones híbridas basadas en un libro semilla,
    etiquetando el nivel de la cascada del cual proviene cada recomendación.
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

    # ==========================================
    # NIVEL 1: Similitud estricta
    # ==========================================
    # Usamos .copy() para evitar advertencias al agregar la nueva columna
    resultados_n1 = resultados[resultados['Similitud_Texto'] >= 0.17].copy()
    resultados_n1['Nivel'] = 'Nivel 1 (Texto)' # <--- NUEVA ETIQUETA
    
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
    
    # ==========================================
    # NIVEL 2: Keywords
    # ==========================================
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
            df_kws['Nivel'] = 'Nivel 2 (Keywords)' # <--- NUEVA ETIQUETA
            df_kws = df_kws.sort_values(by=['Kw_Match', 'Score_Final'], ascending=[False, False])
            recomendaciones = agregar_faltantes(recomendaciones, df_kws)

    # ==========================================
    # NIVEL 3: Área de Conocimiento
    # ==========================================
    if len(recomendaciones) < top_n and area_semilla != 'General':
        df_area = dataframe[dataframe['Area_Conocimiento'] == area_semilla].copy()
        df_area['Score_Final'] = (df_area['W_Editorial_Norm'] * 0.6) + (df_area['W_Citas_Norm'] * 0.4)
        df_area['Similitud_Texto'] = 0.0
        df_area['Nivel'] = 'Nivel 3 (Área)' # <--- NUEVA ETIQUETA
        df_area = df_area.sort_values(by='Score_Final', ascending=False)
        recomendaciones = agregar_faltantes(recomendaciones, df_area)

    # ==========================================
    # NIVEL 4: Top Prestigio Global (Salvavidas)
    # ==========================================
    if len(recomendaciones) < top_n:
        df_global = dataframe.copy()
        df_global['Score_Final'] = (df_global['W_Editorial_Norm'] * 0.6) + (df_global['W_Citas_Norm'] * 0.4)
        df_global['Similitud_Texto'] = 0.0
        df_global['Nivel'] = 'Nivel 4 (Salvavidas)' # <--- NUEVA ETIQUETA
        df_global = df_global.sort_values(by='Score_Final', ascending=False)
        recomendaciones = agregar_faltantes(recomendaciones, df_global)

    # Retornamos el DataFrame incluyendo la nueva columna 'Nivel'
    columnas_retorno = ['Código del libro', 'Titulo_Final', 'Nivel', 'Similitud_Texto', 'W_Editorial_Norm', 'W_Citas_Norm', 'Score_Final']
    return recomendaciones.head(top_n)[columnas_retorno]

# ==========================================
# FUNCIÓN DE EVALUACIÓN MASIVA
# ==========================================
def evaluar_cascada_automatica(dataframe, matriz_tfidf, n_muestras=100):
    print(f"\nIniciando evaluación automática con {n_muestras} libros aleatorios...")
    inicio = time.time()
    
    # 1. Seleccionar libros al azar (random_state=42 garantiza reproducibilidad)
    muestra_codigos = dataframe.sample(n=n_muestras, random_state=42)['Código del libro'].tolist()
    
    # 2. Diccionario para llevar la cuenta de la activación de cada nivel
    conteo_niveles = {
        'Nivel 1 (Texto)': 0,
        'Nivel 2 (Keywords)': 0,
        'Nivel 3 (Área)': 0,
        'Nivel 4 (Salvavidas)': 0
    }
    
    total_recomendaciones = 0
    
    # 3. Iterar sobre la muestra
    for codigo in muestra_codigos:
        resultados = recomendar_por_libro(codigo, dataframe, matriz_tfidf)
        if isinstance(resultados, str): continue # Ignorar errores
            
        conteos_locales = resultados['Nivel'].value_counts().to_dict()
        for nivel, cantidad in conteos_locales.items():
            if nivel in conteo_niveles:
                conteo_niveles[nivel] += cantidad
                
        total_recomendaciones += len(resultados)
        
    tiempo_total = time.time() - inicio
    
    # 4. Calcular porcentajes y formatear la salida
    print(f"Evaluación finalizada en {tiempo_total:.2f} segundos.")
    print(f"Total de libros semilla evaluados: {len(muestra_codigos)}")
    print(f"Total de recomendaciones generadas: {total_recomendaciones}\n")
    
    tabla_resultados = []
    for nivel in ['Nivel 1 (Texto)', 'Nivel 2 (Keywords)', 'Nivel 3 (Área)', 'Nivel 4 (Salvavidas)']:
        cantidad = conteo_niveles[nivel]
        porcentaje = (cantidad / total_recomendaciones) * 100 if total_recomendaciones > 0 else 0
        tabla_resultados.append([nivel, cantidad, f"{porcentaje:.2f}%"])
        
    print("RESUMEN DE TASA DE ACTIVACIÓN DE LA CASCADA:")
    print(tabulate(tabla_resultados, headers=['Nivel de Activación', 'Recomendaciones Entregadas', 'Porcentaje (%)'], tablefmt='fancy_grid'))
    
    return conteo_niveles

# ==========================================
# ÁREA DE PRUEBAS
# ==========================================
if __name__ == "__main__":
    
    # PRUEBA 1: EVALUACIÓN MASIVA (100 Libros)
    # ---------------------------------------------------------
    resultados_evaluacion = evaluar_cascada_automatica(df, tfidf_matrix, n_muestras=100)
    
    # PRUEBA 2: PRUEBA INDIVIDUAL (Comentada temporalmente)
    # ---------------------------------------------------------
    """
    codigo_prueba = 'UJT0179' 
    libro_info = df[df['Código del libro'] == codigo_prueba].iloc[0]
    print(f"\nLibro Semilla: {libro_info['Titulo_Final']} (Autor: {libro_info['Autor_Final']})")
    print("-" * 100)
    top_10 = recomendar_por_libro(codigo_prueba, df, tfidf_matrix)
    columnas_ver = ['Código del libro', 'Titulo_Final', 'Nivel', 'Similitud_Texto', 'W_Editorial_Norm','W_Citas_Norm', 'Score_Final']
    df_imprimir = top_10[columnas_ver].copy()
    df_imprimir['Titulo_Final'] = df_imprimir['Titulo_Final'].apply(lambda x: x[:45] + '...' if len(str(x)) > 45 else x)
    df_imprimir['Similitud_Texto'] = df_imprimir['Similitud_Texto'].round(3)
    df_imprimir['Score_Final'] = df_imprimir['Score_Final'].round(3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_imprimir.to_string(index=False, justify='right'))
    """