import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


iris = load_iris()
X = StandardScaler().fit_transform(iris.data)


inertia = []
sil = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, labels))


km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)

print("Optimal clusters: 3")
print("Silhouette Score:", silhouette_score(X, labels))
print("Cluster sizes:", pd.Series(labels).value_counts().sort_index().to_dict())


P = PCA(n_components=2).fit_transform(X)
plt.scatter(P[:, 0], P[:, 1], c=labels, cmap="viridis")
plt.title("K-Means Clustering")
plt.show()

plt.plot(range(2, 11), inertia, "o-")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
