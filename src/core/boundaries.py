"""
boundaries.py

Soft boundary forces for the Semantic Turing Field.

Particles outside the simulation radius are gently pulled
back toward the center instead of hitting an invisible wall.
"""

import numpy as np


def boundary_force(
    positions,
    radius=5.0,
    strength=0.20,
):
    """
    Compute restoring forces for particles
    outside the simulation radius.

    Parameters
    ----------
    positions : ndarray (N,2)

    radius : float
        Radius of the simulation.

    strength : float
        Spring strength.

    Returns
    -------
    ndarray (N,2)
    """

    forces = np.zeros_like(positions)

    distances = np.linalg.norm(
        positions,
        axis=1,
    )

    finite = np.isfinite(distances)

    outside = (distances > radius) & finite

    if not np.any(outside):
        return forces

    direction = -positions[outside] / distances[outside][:, None]

    magnitude = (distances[outside] - radius) * strength

    forces[outside] = direction * magnitude[:, None]

    return forces
