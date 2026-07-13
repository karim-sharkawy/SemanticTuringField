from typing import Dict, Optional

import numpy as np

from src.core.boundaries import boundary_force
from src.core.forces import compute_forces
from src.core.gravity_wave import gravity_wave


class STFSimulation:
    def __init__(
        self,
        similarity_matrix: np.ndarray,
        alpha: float = 0.95,
        beta: float = 0.2,
        dt: float = 0.05,
        damping: float = 0.99,
    ) -> None:
        self.S: np.ndarray = similarity_matrix

        self.alpha: float = alpha
        self.beta: float = beta
        self.dt: float = dt
        self.damping: float = damping

        self.N: int = len(similarity_matrix)

        self.pos: np.ndarray = np.random.rand(self.N, 2) * 2.0
        self.vel: np.ndarray = np.zeros_like(self.pos)

    def step(
        self,
        sentence: Optional[str] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        vecs: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        # original semantic force
        F: np.ndarray = compute_forces(self.pos, self.S, self.alpha, self.beta)
        F += boundary_force(self.pos)

        # gravity wave
        if sentence is not None and embeddings is not None and vecs is not None:
            F += gravity_wave(sentence, embeddings, vecs, self.pos)

        # full force
        self.vel += F * self.dt
        self.pos += self.vel * self.dt

        self.vel *= self.damping

        return self.pos
