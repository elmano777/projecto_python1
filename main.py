import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Cargar los datos del archivo CSV
df = pd.read_csv('smogon.csv')

# Seleccionar la columna de texto para el análisis tf-idf
corpus = df['moves']

# Definir el vectorizador tf-idf con n-gramas (en este ejemplo, unigramas y bigramas)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Calcular la matriz tf-idf
tfidf_matrix = vectorizer.fit_transform(corpus)

# Mostrar el número total de tokens
total_tokens = len(vectorizer.get_feature_names_out())
print("Número total de tokens en la matriz tf-idf:", total_tokens)

# Imprimir todos los tokens
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
tfidf_df.to_csv('tfidf_clusters.csv', index=False)

# Interpretar los clusters
cluster_labels = {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5'}
for cluster_id, label in cluster_labels.items():
    cluster_indices = tfidf_df.index[tfidf_df['Cluster'] == cluster_id]
    print(f"{label}: {len(cluster_indices)} documentos")