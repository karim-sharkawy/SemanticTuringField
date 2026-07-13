import numpy as np


# Gentle restoring force toward the origin once particles leave the simulation radius.
# Looks better than an invisible wall
def boundary_force(pos: np.ndarray, radius: float = 5.0, strength: float = 0.15) -> np.ndarray:
    F: np.ndarray = np.zeros_like(pos)

    distances: np.ndarray = np.linalg.norm(pos, axis=1)

    outside: np.ndarray = distances > radius

    if np.any(outside):
        directions: np.ndarray = -pos[outside] / distances[outside][:, None]

        magnitudes: np.ndarray = strength * (distances[outside] - radius)

        F[outside] = directions * magnitudes[:, None]

    return F
