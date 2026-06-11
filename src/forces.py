import numpy as np

def compute_forces(pos, similarity_matrix, alpha, beta):
    F = np.zeros_like(pos)

    for i in range(len(pos)):
        diff = pos[i] - pos

        strength = -alpha * (
            similarity_matrix[i] - beta
        )

        F[i] = np.sum(
            strength[:, None] * diff,
            axis=0
        )

    return F
