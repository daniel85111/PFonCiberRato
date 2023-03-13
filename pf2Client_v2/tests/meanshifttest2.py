import numpy as np
import cv2
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

# Visualizando os clusters
canvas = np.zeros((600, 600, 3), dtype=np.uint8)

colors = np.random.randint(0, 255, (n_clusters, 3))

for i in range(X.shape[0]):
    cluster_idx = labels[i]
    color = tuple(map(int, colors[cluster_idx]))
    center = tuple((X[i] + 3) * 100)
    center = np.array(center, dtype=int)
    cv2.circle(canvas, tuple(center), 5, color, -1)

cv2.imshow("Clusters", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
