import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Cargar el archivo CSV
smogon3 = pd.read_csv('smogon.csv')

# Lista de tipos de Pokémon
pokemon_types = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground',
    'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
]

# Función para procesar la columna 'moves'
def filter_pokemon_types(text, types):
    # Convertir el texto a minúsculas y dividir en palabras
    words = re.findall(r'\b\w+\b', text.lower())
    # Mantener solo las palabras que son exactamente iguales a un tipo de Pokémon
    filtered_words = [word for word in words if word in types]
    return ' '.join(filtered_words)

# Aplicar la función a la columna 'moves'
smogon3['filtered_moves'] = smogon3['moves'].apply(lambda x: filter_pokemon_types(str(x), pokemon_types))

# Mostrar el DataFrame después de filtrar los tipos de Pokémon en la columna 'moves'
print("DataFrame después de filtrar los tipos de Pokémon en la columna 'moves':")
print(smogon3[['moves', 'filtered_moves']].head())

# Seleccionar la columna de texto filtrado para el análisis tf-idf
corpus = smogon3['filtered_moves']

# Definir el vectorizador tf-idf usando unigramas
vectorizer = TfidfVectorizer(ngram_range=(1, 1))

# Calcular la matriz tf-idf
tfidf_matrix = vectorizer.fit_transform(corpus)

# Mostrar el número total de tokens
total_tokens = len(vectorizer.get_feature_names_out())
print("Número total de tokens en la matriz tf-idf:", total_tokens)

# Imprimir todos los tokens (elementos de vocabulario)
print("Todos los tokens (elementos de vocabulario):", vectorizer.get_feature_names_out())

# Generar un DataFrame con la matriz tf-idf
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Imprimir la matriz tf-idf
print("Matriz tf-idf:")
print(tfidf_df)

# Realizar el agrupamiento con KMeans
n_clusters = 5  # Seleccionar el número de clusters deseado
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(tfidf_matrix)

# Agregar los clusters al DataFrame
tfidf_df['Cluster'] = clusters

# Generar un archivo CSV con la matriz tf-idf y el cluster
tfidf_df.to_csv("filtered_tfidf_clusters.csv", index=False)
print("Archivo CSV guardado en: filtered_tfidf_clusters.csv")

# Interpretar los clusters
cluster_labels = {i: f'Cluster {i+1}' for i in range(n_clusters)}
for cluster_id, label in cluster_labels.items():
    cluster_indices = tfidf_df.index[tfidf_df['Cluster'] == cluster_id]
    print(f"{label}: {len(cluster_indices)} documentos")
