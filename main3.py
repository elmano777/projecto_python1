import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Cargar el archivo CSV
smogon3 = pd.read_csv('smogon.csv')

# Lista de tipos de Pokémon
tipos = [
    'normal', 'fuego', 'agua', 'eléctrico', 'planta', 'hielo', 'lucha', 'veneno', 'tierra',
    'volador', 'psíquico', 'bicho', 'roca', 'fantasma', 'dragón', 'siniestro', 'acero', 'hada'
]

# Función para procesar la columna 'moves'
def filtrar_tipos_pokemon(texto, tipos):
    # Convertir el texto a minúsculas y dividir en palabras
    palabras = re.findall(r'\b\w+\b', texto.lower())
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
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(matriz_tfidf)

# Agregar los clusters al DataFrame
tfidf['Cluster'] = clusters

# Generar un archivo CSV con la matriz TF-IDF y el cluster
tfidf.to_csv("clusters_tfidf_filtrados.csv", index=False)
print("Archivo CSV guardado en: clusters_tfidf_filtrados.csv")

# Interpretar los clusters
cluster_labels = {i: f'Cluster {i+1}' for i in range(n_clusters)}
for id_cluster, labels in cluster_labels.items():
    indices_cluster = tfidf.index[tfidf['Cluster'] == id_cluster]
    print(f"{labels}: {len(indices_cluster)} documentos")
