import numpy as np

from config import *

from src.load_embeddings import load_embeddings

from src.semantics import lower_dimensions, build_similarity_matrix

from src.simulate_engine import STFSimulation

from src.clustering import cluster_positions
from src.plotting import plot_positions


def main():

    embeddings = load_embeddings(
        "data/glove.2024.wikigiga.50d_small.txt",
        max_words=MAX_WORDS
    )

    words = list(embeddings.keys())

    vecs = np.array(
        list(embeddings.values())
    )

    vecs = lower_dimensions(vecs)

    S = build_similarity_matrix(vecs)

    simulation = STFSimulation(
        S,
        alpha=ALPHA,
        beta=BETA,
        dt=DT,
        damping=DAMPING
    )

    clusters = None

    for step in range(NUM_STEPS):

        pos = simulation.step()

        if step % 200 == 0:
            clusters = cluster_positions(
                pos,
                NUM_CLUSTERS
            )

        if step % 100 == 0:
            plot_positions(
                pos,
                clusters,
                step
            )


if __name__ == "__main__":
    main()