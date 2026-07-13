import numpy as np

from scripts.load_embeddings import load_embeddings
from src.clustering import cluster_embeddings
from src.config import *
from src.plotting import plot_positions
from src.semantics import build_similarity_matrix, lower_dimensions
from src.simulate_engine import STFSimulation


def main():

    try:
        embeddings = np.load(
            EMBEDDINGS_PATH, allow_pickle=True
        ).item()  # item() loads it as original, a dictionairy in this case
    except FileNotFoundError:
        embeddings = load_embeddings(DATA_PATH, MAX_WORDS)
        np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=True)

    words = list(embeddings.keys())

    vecs = np.array(list(embeddings.values()))

    vecs = lower_dimensions(vecs)

    S = build_similarity_matrix(vecs)

    simulation = STFSimulation(S, alpha=ALPHA, beta=BETA, dt=DT, damping=DAMPING)

    clusters, labels = None, None
    for step in range(NUM_STEPS):
        pos = simulation.step()
        """ will be used once i start working on and testing inputs
        user_sentence = input("Sentence: ")

        pos = simulation.step(
            sentence=user_sentence,
            embeddings=embeddings,
            vecs=vecs,
        )
        """

        if step % 200 == 0:
            clusters, labels = cluster_embeddings(vecs, words, NUM_CLUSTERS)

        if step % 100 == 0:
            plot_positions(pos, clusters, labels, step)


if __name__ == "__main__":
    main()
