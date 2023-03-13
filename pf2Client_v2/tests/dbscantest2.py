import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import cv2

# Criando um conjunto de dados aleatório com 100 amostras e 4 clusters
X, y = make_blobs(n_samples=100, centers=4, random_state=42)

# Instanciando um objeto DBSCAN com eps=0.5 e min_samples=5
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Executando o algoritmo DBSCAN no conjunto de dados
dbscan.fit(X)

# Definindo um conjunto de cores para cada rótulo de cluster
colors = [
    (255, 0, 0),  # vermelho
    (0, 255, 0),  # verde
    (0, 0, 255),  # azul
    (255, 255, 0),  # amarelo
    (255, 0, 255),  # magenta
    (0, 255, 255),  # ciano
]

# Criando uma matriz de zeros do mesmo tamanho do conjunto de dados
height, width = X.max(axis=0) + 50
image = np.zeros((int(height), int(width), 3), dtype=np.uint8)

# Atribuindo a cada ponto do conjunto de dados uma cor correspondente ao rótulo do cluster a que pertence
for i, point in enumerate(X):
    label = dbscan.labels_[i]
    if label == -1:
        color = (0, 0, 0)  # preto para pontos de ruído
    else:
        color = colors[label % len(colors)]
    cv2.circle(image, tuple(point.astype(int)), 5, color, -1)

# Exibindo a matriz como uma imagem usando o OpenCV
cv2.imshow('DBSCAN Clustering', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

