import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import threading
import time

def load_embeddings(path, max_words=500):
    embeddings = {} # {"word": np.array([...]), ...}

    with open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            if index >= max_words:
                break
            split_line = line.strip().split()
            word = split_line[0]
            word_embedding=np.array(split_line[1:], dtype=np.float64)
            embeddings[word] = word_embedding

        return embeddings

glove_embeddings = load_embeddings("data/glove.2024.wikigiga.50d_small.txt", max_words=500)
words, vecs = list(glove_embeddings.keys()), np.array(list(glove_embeddings.values()))
word_to_idx = {word: idx for idx, word in enumerate(words)} # for O(1) lookup

# use PCA to lower dimensions down to 50, for speed
def lower_dimensions(vecs):
    if vecs.shape[1] > 50:
        print(f"Reducing dimensions from {vecs.shape[1]} to 50...")
        pca = PCA(n_components=50)
        vecs = pca.fit_transform(vecs)
    return vecs
vecs = lower_dimensions(vecs)

# find similarity between words
print("Computing similarity matrix...")
S = cosine_similarity(vecs) # N x N similarity matrix: words x context
S = (S - S.mean()) / S.std() # normalize to z-scores

#particle setup
N = len(words) # of course limited by max_words, but good to have
pos = np.random.rand(N,2) * 2.0 # position (x,y)
vel = np.zeros_like(pos) # zero velocity for now
mass = np.ones(N) #figure that out later

# force simulation parameters
alpha=0.95
beta=0.2
dt=0.05
damping=0.99

## Force Computation
def compute_forces(pos, S, alpha, beta):
    F = np.zeros_like(pos)

    for i in range(len(pos)): # len(pos) == len(words)
        diff = pos[i] - pos # Shape: (N, 2) - diff to all other words
        strength = -alpha * (S[i] - beta) #attract/repel
        F[i] = np.sum(strength[:, None] * diff, axis=0)
    return F

# main()
for step in range(1000):
    F = compute_forces(pos, S, alpha, beta)
    vel += F *dt
    pos += vel*dt
    vel *= damping

    pos = np.clip(pos, -5, 5) #soft boundary

    if step % 100 == 0:
        plt.clf()

        if step % 200 == 0:
            kmeans = KMeans(n_clusters=8, random_state=0, n_init=10)
            clusters = kmeans.fit_predict(pos) # colors on map indicate words in the same cluster!

        plt.scatter(pos[:, 0], pos[:, 1], c=clusters, s=8, cmap='tab10', alpha=0.7)
        plt.title(f"Step {step}")
        plt.pause(0.01)