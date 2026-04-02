# -*- coding: utf-8 -*-
"""
limpieza.py
===========
Ejecuta la limpieza (Sprint 2) sobre el catálogo unificado de OpenAlex.
- Corrige áreas de conocimiento (BISAC).
- Filtra falsos positivos en la similitud de títulos (Zona Gris).
- Consolida campos clave y construye el 'Tag' (Corpus NLP) para el recomendador.
"""

import re
import logging
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

import nltk
from nltk.corpus import stopwords

# Descarga segura de stopwords de NLTK para los 4 idiomas de la base
try:
    stop_words_es = set(stopwords.words('spanish'))
    stop_words_en = set(stopwords.words('english'))
    stop_words_fr = set(stopwords.words('french'))
    stop_words_pt = set(stopwords.words('portuguese'))
except LookupError:
    nltk.download('stopwords')
    stop_words_es = set(stopwords.words('spanish'))
    stop_words_en = set(stopwords.words('english'))
    stop_words_fr = set(stopwords.words('french'))
    stop_words_pt = set(stopwords.words('portuguese'))

# Unimos todas las palabras vacías en un solo gran filtro
STOPWORDS_COMBINADAS = stop_words_es | stop_words_en | stop_words_fr | stop_words_pt

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
ARCHIVO_ENTRADA = r"data\Libros_Unificados_Recomendador.csv"
ARCHIVO_SALIDA  = r"data\Libros_Limpios_Recomendador.xlsx"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# DICCIONARIOS
# =============================================================================
DICCIONARIO_BISAC = {
    'POL': 'Ciencias Políticas', 'POE': 'Poesía', 'PHI': 'Filosofía',
    'PSY': 'Psicología', 'HIS': 'Historia', 'EDU': 'Educación',
    'REL': 'Religión', 'LAW': 'Derecho', 'BUS': 'Negocios y Economía',
    'ART': 'Arte', 'LIT': 'Literatura', 'SOC': 'Ciencias Sociales',
    'MED': 'Medicina', 'TEC': 'Tecnología', 'SCI': 'Ciencia',
    'TRA': 'Viajes y Transporte', 'NAT': 'Naturaleza', 'SPO': 'Deportes',
    'MUS': 'Música', 'YAN': 'No Ficción Juvenil'
}

TRADUCCIONES_DIRECTAS = {
    'LEY': 'Derecho', 'LAW': 'Derecho', 'HISTORY': 'Historia',
    'BUSINESS': 'Negocios y Economía', 'EDUCATION': 'Educación',
    'RELIGION': 'Religión', 'NATURE': 'Naturaleza', 'MEDICAL': 'Medicina',
    'TECHNOLOGY': 'Tecnología', 'ENGINEERING': 'Ingeniería',
    'PHILOSOPHY': 'Filosofía', 'MUSIC': 'Música'
}

# =============================================================================
# FUNCIONES DE LIMPIEZA Y POST-FILTRADO
# =============================================================================

def estandarizar_area(valor: str) -> str:
    if pd.isna(valor): return 'General'
    valor_str = str(valor).upper().strip()
    match = re.search(r'([A-Z]{3})\d{2,}', valor_str)
    if match: return DICCIONARIO_BISAC.get(match.group(1), 'General')
    for clave, traduccion in TRADUCCIONES_DIRECTAS.items():
        if clave in valor_str: return traduccion
    texto_limpio = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', valor_str).strip()
    return texto_limpio.capitalize() if texto_limpio else 'General'

def limpiar_falsos_positivos(fila: pd.Series) -> pd.Series:
    metodo = str(fila['Metodo_Match'])
    score = fila.get('OpenAlex_Similarity', 0.0)
    tit_local = str(fila['Titulo']).lower().strip()
    tit_oa = str(fila.get('OpenAlex_Title', '')).lower().strip()
    
    if metodo not in ['Solo Titulo', 'Titulo+Autor'] or score >= 0.85:
        return fila
        
    tit_local_clean = re.sub(r'[^\w\s]', '', tit_local)
    tit_oa_clean = re.sub(r'[^\w\s]', '', tit_oa)
    
    if tit_local_clean in tit_oa_clean: return fila 
        
    similitud_parcial = SequenceMatcher(None, tit_local_clean, tit_oa_clean[:len(tit_local_clean) + 5]).ratio()
    if similitud_parcial >= 0.85: return fila 
        
    fila['Metodo_Match'] = 'No Encontrado'
    fila['OpenAlex_Similarity'] = 0.0
    for col in ['OpenAlex_ID', 'OpenAlex_Title', 'OpenAlex_Author', 'OpenAlex_Concepts']:
        fila[col] = None
    fila['OpenAlex_Citations'] = 0
    fila['W_Citas'] = 0.0 
    return fila

# =============================================================================
# CONSTRUCCIÓN DEL TAG NLP
# =============================================================================

def limpiar_titulo_biblioteca(titulo):
    if pd.isna(titulo): return ""
    t = str(titulo).strip()
    t = re.sub(r'\(.*?\)', '', t)
    t = re.sub(r'(.*),\s*(LA|EL|LOS|LAS|UN|UNA|UNOS|UNAS)$', r'\2 \1', t, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', t).strip()

def limpiar_autor_biblioteca(autor: str) -> str:
    if pd.isna(autor): return ""
    a = str(autor).strip()

    # 1. Eliminar prefijos institucionales o ruidosos
    prefijos_ruido = [r'^/?\s*Comité editorial[,\s]*', r'^A:\s*', r'^E:\s*']
    for p in prefijos_ruido:
        a = re.sub(p, '', a, flags=re.IGNORECASE)

    # 2. Manejar "Varios autores" cuando está mezclado con nombres reales
    # Ej: "Varios Autores; Aquiles Omar..." -> Elimina "Varios Autores;"
    a_sin_varios = re.sub(r'(?i)varios autores\s*[;,]*', '', a).strip()
    
    # Si después de borrar "varios autores" quedó vacío, lo devolvemos tal cual 
    # para que la función de consolidación lo detecte y use el de OpenAlex.
    if not a_sin_varios:
        return "Varios autores"
    
    # 3. Extracción del primer autor real
    # Primero dividimos por punto y coma (;)
    partes = [p.strip() for p in a_sin_varios.split(';') if p.strip()]
    if partes:
        primer_bloque = partes[0]
        # Si el primer bloque tiene varios autores separados por coma (,)
        # Ej: "Adriana Camacho-Ramírez, María Catalina Romero" -> Toma "Adriana"
        primer_autor = primer_bloque.split(',')[0].strip()
        return primer_autor

    return ""

def consolidar_textos(fila: pd.Series) -> pd.Series:
    """Prioriza datos locales limpios, pero usa OpenAlex si el local es inválido."""
    # Consolidar Título
    tit_local = limpiar_titulo_biblioteca(fila.get('Titulo'))
    tit_oa = str(fila.get('OpenAlex_Title', '')).strip()
    fila['Titulo_Final'] = tit_local if tit_local else tit_oa

    # Consolidar Autor
    aut_local = limpiar_autor_biblioteca(fila.get('Autor'))
    aut_oa = str(fila.get('OpenAlex_Author', '')).strip()
    autores_invalidos = ['sin datos', 'varios autores', 'nan', '']
    
    if aut_local.lower() in autores_invalidos and aut_oa and aut_oa != 'nan':
        fila['Autor_Final'] = aut_oa
    else:
        fila['Autor_Final'] = aut_local

    return fila

def limpiar_texto_nlp(texto: str) -> str:
    """Aplica técnicas NLP: minúsculas, remoción de signos y stopwords."""
    if pd.isna(texto): return ""
    texto = str(texto).lower()
    texto = re.sub(r'[^\w\s]', ' ', texto) # Quita puntuación
    palabras = texto.split()
    # Filtra stopwords y palabras de 1 o 2 letras (ej: "a", "en")
    palabras_utiles = [p for p in palabras if p not in STOPWORDS_COMBINADAS and len(p) > 2]
    return " ".join(palabras_utiles)

def construir_tag(fila: pd.Series) -> str:
    # Función auxiliar para atrapar nulos antes de convertirlos a texto
    def texto_seguro(columna):
        valor = fila.get(columna)
        return "" if pd.isna(valor) else str(valor).strip()

    # 1. Título doble
    titulo = texto_seguro('Titulo_Final')
    titulo_peso = f"{titulo} {titulo}" if titulo else ""
    
    # 2. Keywords triple
    keywords = texto_seguro('Keywords').replace(';', ' ')
    keywords_peso = f"{keywords} {keywords} {keywords}" if keywords else ""
    
    # 3. Editorial doble como Token Único
    editorial = texto_seguro('Editorial').replace(' ', '') 
    editorial_peso = f"{editorial} {editorial}" if editorial else ""

    elementos = [
        titulo_peso,
        texto_seguro('Autor_Final'),
        editorial_peso,
        keywords_peso,
        texto_seguro('Area_Conocimiento'),
        texto_seguro('OpenAlex_Concepts').replace(';', ' '),
        texto_seguro('Abstract')
    ]
    
    # Unimos solo los elementos que realmente tengan texto (ignorando los vacíos)
    texto_crudo = " ".join([elem for elem in elementos if elem.strip()])
    return limpiar_texto_nlp(texto_crudo)

# =============================================================================
# PUNTO DE ENTRADA (PIPELINE)
# =============================================================================

def main() -> None:
    log.info(f"Cargando datos desde '{ARCHIVO_ENTRADA}'...")
    df = pd.read_csv(ARCHIVO_ENTRADA)
    
    log.info("1. Estandarizando las Áreas de Conocimiento...")
    df['Area_Conocimiento'] = df['BISAC Catálogo'].apply(estandarizar_area)

    log.info("1.1 Recalculando prestigio editorial con áreas unificadas...")
    
    # Borramos los cálculos viejos para no generar conflicto
    columnas_viejas = ['Titulos_Editorial_Area', 'W_Editorial']
    df = df.drop(columns=[c for c in columnas_viejas if c in df.columns])

    prestigio = (
    df.groupby(["Editorial", "Area_Conocimiento"])
    .size()
    .reset_index(name="Titulos_Editorial_Area")
    )

    df = df.merge(prestigio, on=["Editorial", "Area_Conocimiento"], how="left")

    #Fórmula de prestigio editorial: 1 + logaritmo del número de títulos en esa área para esa editorial
    df["W_Editorial"] = 1 + (0.1 * np.log1p(df["Titulos_Editorial_Area"]))

    log.info("2. Purgando falsos positivos de OpenAlex (Zona Gris)...")
    df = df.apply(limpiar_falsos_positivos, axis=1)
    
    log.info("3. Consolidando Títulos y Autores...")
    df = df.apply(consolidar_textos, axis=1)
    
    log.info("4. Construyendo el 'Tag' semántico (NLP)...")
    df['Tag'] = df.apply(construir_tag, axis=1)
    
    # Opcional: Eliminar columnas crudas que ya no se usarán para aligerar la base final
    columnas_a_borrar = ['BISAC Catálogo', 'Thema Catálogo', 'Clasificación Dewey', 'Abstract', 'Keywords', 'OpenAlex_Concepts']
    df_final = df.drop(columns=[col for col in columnas_a_borrar if col in df.columns])
    
    log.info(f"Guardando base lista para el modelo en '{ARCHIVO_SALIDA}'...")
    df_final.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")
    
    log.info("¡Proceso de limpieza finalizado con éxito!")

if __name__ == "__main__":
    main()# -*- coding: utf-8 -*-
