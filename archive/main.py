from typing import Dict, Optional

import numpy as np
from src.visualization.clustering import cluster_embeddings
from src.visualization.plotting import plot_positions

from src.core.simulate_engine import STFSimulation
from src.nlp.embeddings import load_embeddings
from src.nlp.semantics import build_similarity_matrix, lower_dimensions
from src.utils.config import *


def main() -> None:
    current_sentence: Optional[str] = None
    gravity_steps_remaining = 0

    try:
        embeddings: Dict[str, np.ndarray] = np.load(
            EMBEDDINGS_PATH, allow_pickle=True
        ).item()  # item() loads it as original, a dictionairy in this case
    except FileNotFoundError:
        embeddings = load_embeddings(DATA_PATH, MAX_WORDS)
        np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=True)

    words: list[str] = list(embeddings.keys())

    vecs: np.ndarray = np.array(list(embeddings.values()))

    vecs = lower_dimensions(vecs)

    S = build_similarity_matrix(vecs)

    simulation = STFSimulation(S, alpha=ALPHA, beta=BETA, dt=DT, damping=DAMPING)

    clusters: Optional[np.ndarray] = None
    labels: Optional[list[str]] = None
    for step in range(NUM_STEPS):
        if gravity_steps_remaining == 0:
            user_sentence: str = input("Sentence (Enter to skip): ").strip()

            if user_sentence:
                current_sentence = user_sentence
                gravity_steps_remaining = 75

        if gravity_steps_remaining > 0:
            pos = simulation.step(
                sentence=current_sentence,
                embeddings=embeddings,
                vecs=vecs,
            )
            gravity_steps_remaining -= 1
        else:
            pos = simulation.step()

        if step % 200 == 0:
            clusters, labels = cluster_embeddings(vecs, words, NUM_CLUSTERS)

        if step % 100 == 0:
            plot_positions(pos, clusters, labels, step)


if __name__ == "__main__":
    main()
