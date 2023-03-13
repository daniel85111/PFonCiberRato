import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# Gerando dados aleatórios
np.random.seed(0)
X = np.random.randn(100, 2)

# Estimando a largura de banda usando o método de Scott
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=20)

# Criando o modelo de clustering por meanshift
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# Fit do modelo aos dados
ms.fit(X)

# Obtendo as labels de cada ponto
labels = ms.labels_

# Obtendo os centróides de cada cluster
centroids = ms.cluster_centers_

# Obtendo o número de clusters
n_clusters = len(np.unique(labels))

print("Número de clusters:", n_clusters)
print("Centróides dos clusters:", centroids)
print("Labels dos pontos:", labels)