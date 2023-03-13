import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Criando um conjunto de dados aleatório com 100 amostras e 4 clusters
X, y = make_blobs(n_samples=100, centers=4, random_state=42)

# Instanciando um objeto DBSCAN com eps=0.5 e min_samples=5
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Executando o algoritmo DBSCAN no conjunto de dados
dbscan.fit(X)

# Imprimindo os rótulos de cluster encontrados pelo DBSCAN
print(dbscan.labels_)
