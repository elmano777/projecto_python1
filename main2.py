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
resultados_pca = pca.fit_transform(smogon2)

# Imprimir el número de filas y columnas del DataFrame original
print("Número de filas y columnas del DataFrame original:", smogon2.shape)

# Imprimir el número de filas y columnas de la matriz de componentes principales
print("Número de filas y columnas de la matriz de componentes principales:", resultados_pca.shape)

# Generar un nuevo DataFrame con la matriz de componentes principales
df_pca = pd.DataFrame(data=resultados_pca, columns=['PCA1', 'PCA2', 'PCA3'])
print("Matriz de componentes principales:")
print(df_pca)

# Realizar el agrupamiento con KMeans en las componentes principales
n_clusters_pca = 5
kmeans_pca = KMeans(n_clusters=n_clusters_pca)
clusters_pca = kmeans_pca.fit_predict(resultados_pca)

# Agregar los clusters al DataFrame de componentes principales
df_pca['Cluster'] = clusters_pca

# Generar un archivo CSV con la matriz de componentes principales y el cluster
df_pca.to_csv('pca_clusters.csv', index=False)
print("Archivo CSV guardado en: pca_clusters.csv")

# Interpretar los clusters
cluster_labels_pca = {i: f'Cluster {i+1}' for i in range(n_clusters_pca)}
for id_cluster, labels in cluster_labels_pca.items():
    indices_cluster_pca = df_pca.index[df_pca['Cluster'] == id_cluster]
    print(f"{labels}: {len(indices_cluster_pca)} documentos")
