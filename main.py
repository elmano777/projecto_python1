import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Cargar los datos del archivo CSV
smogon = pd.read_csv('smogon.csv')
analizar = smogon['moves']

# Generar la matriz TF-IDF utilizando una cantidad de n-gramas elegida por usted (unigramas y bigramas en este caso)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(analizar)

# Mostrar el número total de tokens (elementos de su vocabulario) que tiene su matriz tf-idf
total_tokens = len(vectorizer.get_feature_names_out())
print("Número total de tokens en la matriz tf-idf:", total_tokens)

# Imprimir todos los tokens (elementos de su vocabulario)
print("Todos los tokens (elementos de vocabulario):", vectorizer.get_feature_names_out())

# Generar un DataFrame con la matriz tf-idf que tenga como cabeceras los elementos de su vocabulario
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Imprimir la matriz tf-idf
print("Matriz tf-idf:")
print(tfidf_df)

# Agrupar las filas de su nuevo DataFrame, en base a sus puntuaciones tf-idf (se elige la cantidad de clusters, en este caso 5)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(tfidf_matrix)

# Agregar los clusters al DataFrame
tfidf_df['Cluster'] = clusters

# Generar un archivo de valores separado por comas (CSV) que contenga su matriz tfidf y el cluster
tfidf_df.to_csv('tfidf_clusters.csv', index=False)
print("Archivo CSV guardado en: 'tfidf_clusters.csv'")

# Interpretar los clusters y ponerle un nombre a cada uno
cluster_labels = {i: f'Cluster {i+1}' for i in range(n_clusters)}
for cluster_id, label in cluster_labels.items():
    cluster_indices = tfidf_df.index[tfidf_df['Cluster'] == cluster_id]
    print(f"{label}: {len(cluster_indices)} documentos")


