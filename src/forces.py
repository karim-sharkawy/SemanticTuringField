import numpy as np
from numba import njit

@njit(cache=True)
def compute_forces(pos, similarity_matrix, alpha, beta):
    F = np.zeros_like(pos)

    for i in range(len(pos)):  # len(pos) == len(words)
        diff = pos[i] - pos  # Shape: (N, 2) - diff to all other words

        strength = -alpha * (
            similarity_matrix[i] - beta  # attract/repel
        )

        F[i] = np.sum(strength[:, None] * diff, axis=0)

    return F
