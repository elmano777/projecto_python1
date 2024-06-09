import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Cargar los datos del archivo CSV
smogon = pd.read_csv('smogon.csv')
analizar = smogon['moves']

# Generar la matriz TF-IDF utilizando una cantidad de n-gramas elegida por usted (unigramas y bigramas en este caso)
vectorizador = TfidfVectorizer(ngram_range=(1, 2))
matriz_tfidf = vectorizador.fit_transform(analizar)

# Mostrar el número total de tokens (elementos de su vocabulario) que tiene su matriz tf-idf
total_tokens = len(vectorizador.get_feature_names_out())
print("Número total de tokens en la matriz tf-idf:", total_tokens)

# Imprimir todos los tokens (elementos de su vocabulario)
print("Todos los tokens (elementos de vocabulario):", vectorizador.get_feature_names_out())

# Generar un DataFrame con la matriz tf-idf que tenga como cabeceras los elementos de su vocabulario
tfidf = pd.DataFrame(matriz_tfidf.toarray(), columns=vectorizador.get_feature_names_out())

# Imprimir la matriz tf-idf
print("Matriz tf-idf:")
print(tfidf)

# Agrupar las filas de su nuevo DataFrame, en base a sus puntuaciones tf-idf (se elige la cantidad de clusters, en este caso 5)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(matriz_tfidf)

# Agregar los clusters al DataFrame
tfidf['Cluster'] = clusters

# Generar un archivo de valores separado por comas (CSV) que contenga su matriz tfidf y el cluster
tfidf.to_csv('tfidf_clusters.csv', index=False)
print("Archivo CSV guardado en: 'tfidf_clusters.csv'")

# Interpretar los clusters y ponerle un nombre a cada uno
cluster_labels = {i: f'Cluster {i+1}' for i in range(n_clusters)}
for id_cluster, labels in cluster_labels.items():
    indices_cluster = df_tfidf.index[df_tfidf['Cluster'] == id_cluster]
    print(f"{labels}: {len(indices_cluster)} documentos")
