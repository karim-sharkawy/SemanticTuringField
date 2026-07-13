from typing import Dict, Optional

import numpy as np

from src.nlp.semantics import compute_sentence_similarities
from src.nlp.text_preprocessing import sentence_to_embedding, tokenize


def gravity_wave(
    sentence: str,
    embeddings: Dict[str, np.ndarray],
    vecs: np.ndarray,
    pos: np.ndarray,
    strength: float = 0.5,
    threshold: float = 0.3,
) -> np.ndarray:

    tokens: list[str] = tokenize(sentence)

    sentence_vec: Optional[np.ndarray] = sentence_to_embedding(tokens, embeddings)

    if sentence_vec is None:
        print("No valid tokens present.")
        return np.zeros_like(pos)

    sims: np.ndarray = compute_sentence_similarities(sentence_vec, vecs)

    gravity_force: np.ndarray = np.zeros_like(pos)

    for i in range(len(pos)):
        if sims[i] > threshold:
            direction: np.ndarray = -pos[i]  # toward origin
            gravity_force[i] += direction * strength * sims[i]

    return gravity_force  # ndarray of shape (N, 2)
