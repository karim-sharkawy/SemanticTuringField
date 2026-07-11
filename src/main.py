import numpy as np

from clustering import cluster_embeddings
from config import *
from load_embeddings import load_embeddings
from plotting import plot_positions
from semantics import build_similarity_matrix, lower_dimensions
from simulate_engine import STFSimulation


def main():

    embeddings = load_embeddings("data/glove.2024.wikigiga.50d_small.txt", max_words=MAX_WORDS)

    words = list(embeddings.keys())

    vecs = np.array(list(embeddings.values()))

    vecs = lower_dimensions(vecs)

    S = build_similarity_matrix(vecs)

    simulation = STFSimulation(S, alpha=ALPHA, beta=BETA, dt=DT, damping=DAMPING)

    clusters, labels = None, None
    for step in range(NUM_STEPS):
        pos = simulation.step()

        if step % 200 == 0:
            clusters, labels = cluster_embeddings(vecs, words, NUM_CLUSTERS)

        if step % 100 == 0:
            plot_positions(pos, clusters, labels, step)


if __name__ == "__main__":
    main()
