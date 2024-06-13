import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Cargar el archivo CSV
smogon3 = pd.read_csv('smogon.csv')

# Lista de tipos de Pokémon
tipos = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground',
    'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
]

# Función para procesar la columna 'moves'
def filtrar_tipos_pokemon(texto, tipos):
    # Convertir el texto a minúsculas y dividir en palabras
    palabras = re.findall(r'\w+', texto.lower())
    # Mantener solo las palabras que son exactamente iguales a un tipo de Pokémon
    tipos_buscados = [palabra for palabra in palabras if palabra in tipos]
    return ' '.join(tipos_buscados)

# Aplicar la función a la columna 'moves'
smogon3['movimientos_filtrados'] = smogon3['moves'].apply(lambda x: filtrar_tipos_pokemon(str(x), tipos))

# Mostrar el DataFrame después de filtrar los tipos de Pokémon en la columna 'moves'
print("DataFrame después de filtrar los tipos de Pokémon en la columna 'moves':")
print(smogon3[['moves', 'movimientos_filtrados']].head())

# Seleccionar la columna de texto filtrado para el análisis TF-IDF
corpus = smogon3['movimientos_filtrados']

# Definir el vectorizador TF-IDF usando unigramas
vectorizador = TfidfVectorizer(ngram_range=(1, 1))

# Calcular la matriz TF-IDF
matriz_tfidf = vectorizador.fit_transform(corpus)

# Mostrar el número total de tokens
total_tokens = len(vectorizador.get_feature_names_out())
print("Número total de tokens en la matriz TF-IDF:", total_tokens)

# Imprimir todos los tokens (elementos de vocabulario)
print("Todos los tokens (elementos de vocabulario):", vectorizador.get_feature_names_out())

# Generar un DataFrame con la matriz TF-IDF
tfidf = pd.DataFrame(matriz_tfidf.toarray(), columns=vectorizador.get_feature_names_out())

# Imprimir la matriz TF-IDF
print("Matriz TF-IDF:")
print(tfidf)

# Realizar el agrupamiento con KMeans
n_clusters = 18  # Seleccionar el número de clusters deseado
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(matriz_tfidf)

# Función para obtener los tokens principales de un centroide
def tokens_principales(centroide, vectorizador, n=5):
    indices = np.argsort(centroide)[::-1][:n]
    tokens = [vectorizador.get_feature_names_out()[i] for i in indices]
    return tokens

# Obtener los centroides de cada cluster
centroides = kmeans.cluster_centers_

# Asignar nombres descriptivos a los clusters
nombres_clusters = {}
for i, centroide in enumerate(centroides):
    tokens = tokens_principales(centroide, vectorizador)
    nombre_cluster = f"Cluster {i+1}: {', '.join(tokens)}"
    nombres_clusters[i] = nombre_cluster

# Agregar los clusters al DataFrame con los nombres descriptivos
tfidf['Cluster'] = [nombres_clusters[cluster] for cluster in clusters]

# Generar un archivo CSV con la matriz TF-IDF y el cluster
tfidf.to_csv("clusters_tfidf2.csv", index=False)
print("Archivo CSV guardado en: clusters_tfidf2.csv")

# Interpretar los clusters
for nombre_cluster, indices_cluster in tfidf.groupby('Cluster')['Cluster']:
    print(f"{nombre_cluster}")