"""
simulate_engine.py

Core physics engine for the Semantic Turing Field.
"""

import numpy as np

from .forces import compute_forces
from .gravity_wave import gravity_wave
from .boundaries import boundary_force


class STFSimulation:

    def __init__(
        self,
        similarity_matrix,
        alpha=0.95,
        beta=0.20,
        dt=0.05,
        damping=0.99,
    ):

        self.S = similarity_matrix

        self.alpha = alpha
        self.beta = beta

        self.dt = dt
        self.damping = damping

        self.N = len(similarity_matrix)

        self.reset()

    def step(
        self,
        sentence=None,
        embeddings=None,
        vecs=None,
    ):
        """
        Advance the simulation by one timestep.
        """

        ### Total Force
        total_force = compute_forces(
            self.pos,
            self.S,
            self.alpha,
            self.beta,
        )

        # Gravity Wave
        if (
            sentence is not None
            and embeddings is not None
            and vecs is not None
        ):

            total_force += gravity_wave(
                sentence,
                embeddings,
                vecs,
                self.pos,
            )

        # Boundary

        total_force += boundary_force(
            self.pos
        )

        # Physics Integration
        self.vel += total_force * self.dt

        self.pos += self.vel * self.dt

        self.vel *= self.damping

        return self.pos

    ### Reset
    def reset(self):
        """
        Reset the particle field.
        """

        self.pos = (
            np.random.rand(
                self.N,
                2,
            )
            * 2.0
        )

        self.vel = np.zeros_like(
            self.pos
        )
