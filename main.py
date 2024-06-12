import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords

# Descargar las stopwords de NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

custom_stop_words = stop_words.union({
    'power', 'accuracy', 'user', 'target', 'pp', 'moves', 'the', 'to', 'is', 'on', 'by', 'pokemon', 'battle',
    'spot', 'singles', 'doubles', 'gen', 'other', 'gens'
})

# Convertir el conjunto de stop words a una lista
custom_stop_words = list(custom_stop_words)

# Cargar los datos del archivo CSV
smogon = pd.read_csv('smogon.csv')
analizar = smogon['moves']

# Generar la matriz TF-IDF utilizando unigramas y bigramas
vectorizador = TfidfVectorizer(ngram_range=(1, 2), stop_words=custom_stop_words, max_df=0.85, min_df=2)
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
n_clusters = 18
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(matriz_tfidf)

# Agregar los clusters al DataFrame
tfidf['Cluster'] = clusters

# Generar un archivo de valores separado por comas (CSV) que contenga su matriz tfidf y el cluster
tfidf.to_csv('tfidf_clusters.csv', index=False)
print("Archivo CSV guardado en: 'tfidf_clusters.csv'")

# Interpretar los clusters
cluster_labels = {i: f'Cluster {i+1}' for i in range(n_clusters)}
for cluster_id in range(n_clusters):
    print(f"\nInterpretando el {cluster_labels[cluster_id]}:")
    cluster_data = tfidf[tfidf['Cluster'] == cluster_id]
    top_terms = cluster_data.drop(columns=['Cluster']).mean().sort_values(ascending=False).head(10)
    print("Términos más comunes:", top_terms.index.tolist())
