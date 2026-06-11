import numpy as np
from src.forces import compute_forces

class STFSimulation:

    def __init__(
        self,
        similarity_matrix,
        alpha=0.95,
        beta=0.2,
        dt=0.05,
        damping=0.99
    ):
        self.S = similarity_matrix

        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.damping = damping

        self.N = len(similarity_matrix)

        self.pos = np.random.rand(self.N, 2) * 2.0
        self.vel = np.zeros_like(self.pos)

    def step(self):

        F = compute_forces(
            self.pos,
            self.S,
            self.alpha,
            self.beta
        )

        self.vel += F * self.dt
        self.pos += self.vel * self.dt

        self.vel *= self.damping

        self.pos = np.clip(
            self.pos,
            -5,
            5
        )

        return self.pos