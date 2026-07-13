import numpy as np

from forces import compute_forces
from gravity_wave import gravity_wave


class STFSimulation:
    def __init__(self, similarity_matrix, alpha=0.95, beta=0.2, dt=0.05, damping=0.99):
        self.S = similarity_matrix

        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.damping = damping

        self.N = len(similarity_matrix)

        self.pos = np.random.rand(self.N, 2) * 2.0
        self.vel = np.zeros_like(self.pos)

    def step(self, sentence=None, embeddings=None, vecs=None):

        # original semantic force
        F = compute_forces(self.pos, self.S, self.alpha, self.beta)

        # gravity wave
        if sentence is not None and embeddings is not None and vecs is not None:
            F += gravity_wave(sentence, embeddings, vecs, self.pos)

        # full force
        self.vel += F * self.dt
        self.pos += self.vel * self.dt

        self.vel *= self.damping

        self.pos = np.clip(self.pos, -5, 5)

        return self.pos
