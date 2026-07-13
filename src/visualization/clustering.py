from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def cluster_embeddings(
    vecs: np.ndarray, words: list[str], n_clusters: int = 8
) -> Tuple[np.ndarray, list[str]]:
    kmeans: KMeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)

    clusters: np.ndarray = kmeans.fit_predict(vecs)
    centroids: np.ndarray = kmeans.cluster_centers_

    labels: list[str] = []

    for centroid in centroids:
        similarities: np.ndarray = cosine_similarity(centroid.reshape(1, -1), vecs)[0]

        closest_idx: int = np.argmax(similarities)
        labels.append(words[closest_idx])

    return clusters, labels
