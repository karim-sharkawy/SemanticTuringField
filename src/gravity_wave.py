import numpy as np

from semantics import compute_sentence_similarities
from text_preprocessing import sentence_to_embedding, tokenize


def gravity_wave(sentence, embeddings, vecs, pos, strength=0.5, threshold=0.3):

    tokens = tokenize(sentence)

    sentence_vec = sentence_to_embedding(tokens, embeddings)

    if sentence_vec is None:
        print("No valid tokens present.")
        return np.zeros_like(pos)

    sims = compute_sentence_similarities(sentence_vec, vecs)

    gravity_force = np.zeros_like(pos)

    for i in range(len(pos)):
        if sims[i] > threshold:
            direction = -pos[i]  # toward origin
            gravity_force[i] += direction * strength * sims[i]

    return gravity_force  # ndarray of shape (N, 2)
