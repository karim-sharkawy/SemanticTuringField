import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# use PCA to lower dimensions down to 50, for speed
def lower_dimensions(vecs: np.ndarray) -> np.ndarray:
    if vecs.shape[1] > 50:
        print(f"Reducing dimensions from {vecs.shape[1]} to 50...")
        pca: PCA = PCA(n_components=50)
        vecs = pca.fit_transform(vecs)
    return vecs


# find similarity between words
def build_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    print("Computing similarity matrix...")

    S: np.ndarray = cosine_similarity(vecs)  # N x N similarity matrix: words x context
    S = (S - S.mean()) / S.std()  # normalize to z-scores

    return S


def compute_sentence_similarities(sentence_embedding: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    return cosine_similarity(sentence_embedding.reshape(1, -1), vecs)[0]
