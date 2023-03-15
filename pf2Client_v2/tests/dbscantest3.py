import numpy as np
from sklearn.cluster import DBSCAN
import cv2

# Gerando um conjunto de dados sintético com 300 amostras e 4 clusters
np.random.seed(0)
centers = [[-1, 1], [1, 1], [1, -1], [-1, -1]]
X = np.empty((0, 2))
for i in range(len(centers)):
    cluster = np.random.randn(100, 2) + centers[i]
    X = np.vstack((X, cluster))

# Instanciando um objeto DBSCAN com eps=0.25
dbscan = DBSCAN(eps=0.3)

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
height, width = X.max(axis=0) + 1
image = np.zeros((int(height * 100), int(width * 100), 3), dtype=np.uint8)

# Atribuindo a cada ponto do conjunto de dados uma cor correspondente ao rótulo do cluster a que pertence
for i, point in enumerate(X):
    label = dbscan.labels_[i]
    if label == -1:
        color = (0, 0, 0)  # preto para pontos de ruído
    else:
        color = colors[label % len(colors)]
    cv2.circle(image, (int(point[0]*100), int(point[1]*100)), 5, color, -1)

# Exibindo a matriz como uma imagem usando o OpenCV
cv2.imshow('DBSCAN Clustering', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
