from sklearn.cluster import KMeans

def cluster_positions(pos, n_clusters=8):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init=10
    )

    return kmeans.fit_predict(pos)