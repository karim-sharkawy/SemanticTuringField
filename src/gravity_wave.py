import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import DT as dt


def gravity_wave(sentence, words, vecs, pos, vel, strength=0.5, radius=3.0):
    sentence_vec = np.mean(
        [vecs[words.index(w)] for w in sentence.lower().split() if w in words], axis=0
    )

    sims = cosine_similarity([sentence_vec], vecs)[0]

    N = len(pos)

    for i in range(N):
        if sims[i] > 0.3:
            direction = -pos[i]
            force_mag = strength * sims[i]
            vel[i] += direction * force_mag * dt

    return vel
