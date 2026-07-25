import numpy as np
from numba import njit


@njit(cache=True)
def compute_forces(
    pos: np.ndarray,
    similarity_matrix: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:

    N = pos.shape[0]

    F = np.zeros_like(pos)

    for i in range(N):
        fx = 0.0
        fy = 0.0

        for j in range(N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]

            distance = np.sqrt(dx * dx + dy * dy) + 1e-8

            direction_x = dx / distance
            direction_y = dy / distance

            strength = -alpha * (similarity_matrix[i, j] - beta)

            fx += strength * direction_x
            fy += strength * direction_y

        F[i, 0] = fx / N
        F[i, 1] = fy / N

    return F
