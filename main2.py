import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Cargar el archivo CSV generado en la Parte 1 del trabajo
smogon2 = pd.read_csv('tfidf_clusters.csv')
datos_url = pd.read_csv("smogon.csv")

# Descartar la columna del cluster
smogon2.drop(columns=["Cluster"], inplace=True)
print("DataFrame después de descartar la columna del cluster:")
print(smogon2)

# Descartar la primera columna para eliminar el doble índice
# smogon2.drop(columns=[smogon2.columns[0]], inplace=True)
# print("DataFrame después de descartar la primera columna:")
# print(smogon2)

# Aplicar Análisis de Componentes Principales (PCA)
pca = PCA(n_components=20)
resultados_pca = pca.fit_transform(smogon2)

# Imprimir el número de filas y columnas del DataFrame original
print("Número de filas y columnas del DataFrame original:", smogon2.shape)

# Imprimir el número de filas y columnas de la matriz de componentes principales
print("Número de filas y columnas de la matriz de componentes principales:", resultados_pca.shape)

# Generar un nuevo DataFrame con la matriz de componentes principales
df_pca = pd.DataFrame(data=resultados_pca, columns=['PCA1', 'PCA2', 'PCA3',"PCA4","PCA5","PCA6","PCA7","PCA8","PCA9","PCA10","PCA11","PCA12","PCA13","PCA14","PCA15","PCA16","PCA17","PCA18","PCA19","PCA20"])
print("Matriz de componentes principales:")
print(df_pca)

# Realizar el agrupamiento con KMeans en las componentes principales
n_clusters_pca = 20
kmeans_pca = KMeans(n_clusters=n_clusters_pca,  n_init=40)
clusters_pca = kmeans_pca.fit_predict(resultados_pca)

# Agregar los clusters al DataFrame de componentes principales
df_pca['Cluster'] = clusters_pca
df_pca["URL"] = datos_url["url"]

# Generar un archivo CSV con la matriz de componentes principales y el cluster
df_pca.to_csv('pca_clusters.csv')
print("Archivo CSV guardado en: pca_clusters.csv")
print(df_pca)