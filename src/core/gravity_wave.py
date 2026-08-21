from typing import Dict, Optional

import numpy as np

from src.nlp.semantics import compute_sentence_similarities
from src.nlp.text_preprocessing import sentence_to_embedding, stopword_removal, tokenize


def gravity_wave(
    sentence: str,
    embeddings: Dict[str, np.ndarray],
    vecs: np.ndarray,
    pos: np.ndarray,
    strength: float = 0.5,
    threshold: float = 0.3,
) -> np.ndarray:

    tokens: list[str] = tokenize(sentence)

    tokens = stopword_removal(tokens)

    # convert the sentence into its average embedding
    sentence_vec: Optional[np.ndarray] = sentence_to_embedding(
        tokens,
        embeddings,
    )

    if sentence_vec is None:
        print("No valid tokens present.")
        return np.zeros_like(pos)

    # compare every word in the field against the meaning of the entire sentence
    sims: np.ndarray = compute_sentence_similarities(
        sentence_vec,
        vecs,
    )

    # pull semantically relevant particles toward the origin
    gravity_force: np.ndarray = -pos * strength * np.maximum(sims - threshold, 0)[:, None]

    return gravity_force
