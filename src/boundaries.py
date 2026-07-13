import numpy as np

# Gentle restoring force toward the origin once particles leave the simulation radius.
# Looks better than an invisible wall
def boundary_force(pos, radius=5.0, strength=0.15):
    F = np.zeros_like(pos)

    distances = np.linalg.norm(pos, axis=1)

    outside = distances > radius

    if np.any(outside):

        directions = -pos[outside] / distances[outside][:, None]

        magnitudes = strength * (distances[outside] - radius)

        F[outside] = directions * magnitudes[:, None]

    return F