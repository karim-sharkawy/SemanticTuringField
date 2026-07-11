import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def cluster_embeddings(vecs, words, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)

    clusters = kmeans.fit_predict(vecs)
    centroids = kmeans.cluster_centers_

    labels = []

    for centroid in centroids:
        similarities = cosine_similarity(centroid.reshape(1, -1), vecs)[0]

        closest_idx = np.argmax(similarities)
        labels.append(words[closest_idx])

    return clusters, labels
