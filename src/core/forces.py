import numpy as np
from numba import njit


@njit(cache=True)
def compute_forces(
    pos: np.ndarray, similarity_matrix: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    F: np.ndarray = np.zeros_like(pos)

    for i in range(len(pos)):  # len(pos) == len(words)
        diff: np.ndarray = pos[i] - pos  # Shape: (N, 2) - diff to all other words

        strength: np.ndarray = -alpha * (
            similarity_matrix[i] - beta  # attract/repel
        )

        F[i] = np.sum(strength[:, None] * diff, axis=0)

    return F
