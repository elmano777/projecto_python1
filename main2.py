import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Cargar el archivo CSV generado en la Parte 1 del trabajo
smogon2 = pd.read_csv('tfidf_clusters.csv')

# Descartar la columna del cluster
smogon2.drop(columns=['Cluster'], inplace=True)
print("DataFrame después de descartar la columna del cluster:")
print(smogon2)

# Descartar la primera columna para eliminar el doble índice
smogon2.drop(columns=[smogon2.columns[0]], inplace=True)
print("DataFrame después de descartar la primera columna:")
print(smogon2)

# Aplicar Análisis de Componentes Principales (PCA)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(smogon2)

# Imprimir el número de filas y columnas del DataFrame original
print("Número de filas y columnas del DataFrame original:", smogon2.shape)

# Imprimir el número de filas y columnas de la matriz de componentes principales
print("Número de filas y columnas de la matriz de componentes principales:", pca_result.shape)

# Generar un nuevo DataFrame con la matriz de componentes principales
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2', 'PCA3'])
print("Matriz de componentes principales:")
print(pca_df)

# Realizar el agrupamiento con KMeans en las componentes principales
n_clusters_pca = 5
kmeans_pca = KMeans(n_clusters=n_clusters_pca)
clusters_pca = kmeans_pca.fit_predict(pca_result)

# Agregar los clusters al DataFrame de componentes principales
pca_df['Cluster'] = clusters_pca

# Generar un archivo CSV con la matriz de componentes principales y el cluster
pca_df.to_csv('pca_clusters.csv', index=False)
print("Archivo CSV guardado en: pca_clusters.csv")

# Interpretar los clusters
cluster_labels_pca = {i: f'Cluster {i+1}' for i in range(n_clusters_pca)}
for cluster_id, label in cluster_labels_pca.items():
    cluster_indices_pca = pca_df.index[pca_df['Cluster'] == cluster_id]
    print(f"{label}: {len(cluster_indices_pca)} documentos")


