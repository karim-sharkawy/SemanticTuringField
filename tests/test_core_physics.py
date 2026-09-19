import numpy as np

from src.core.boundaries import boundary_force
from src.core.forces import compute_forces
from src.core.gravity_wave import gravity_wave
from src.core.simulate_engine import STFSimulation


def test_boundary_force_restores_outside_particles():
    positions = np.array([[0.0, 0.0], [2.0, 0.0], [7.0, 0.0]], dtype=float)

    forces = boundary_force(positions, radius=5.0, strength=0.2)

    expected = np.array([[0.0, 0.0], [0.0, 0.0], [-0.4, 0.0]], dtype=float)
    np.testing.assert_allclose(forces, expected, atol=1e-12)


def test_compute_forces_matches_expected_pairwise_pull():
    positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    similarity_matrix = np.zeros((2, 2), dtype=float)

    forces = compute_forces(positions, similarity_matrix, alpha=0.5, beta=0.2)

    expected = np.array([[-0.05, 0.0], [0.05, 0.0]], dtype=float)
    np.testing.assert_allclose(forces, expected, atol=1e-12)


def test_gravity_wave_pull_toward_origin_for_relevant_tokens():
    positions = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)
    embeddings = {
        "apple": np.array([1.0, 0.0]),
        "banana": np.array([1.0, 0.0]),
    }
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float)

    force = gravity_wave(
        sentence="apple banana",
        embeddings=embeddings,
        vecs=vecs,
        pos=positions,
        strength=0.5,
        threshold=0.3,
    )

    expected = -positions * 0.35
    np.testing.assert_allclose(force, expected, atol=1e-12)


def test_stf_simulation_step_updates_positions_and_counts_steps():
    similarity_matrix = np.zeros((2, 2), dtype=float)
    simulation = STFSimulation(similarity_matrix, alpha=0.5, beta=0.2, dt=0.1, damping=0.9)

    simulation.pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    simulation.vel = np.zeros_like(simulation.pos)

    previous_positions = simulation.pos.copy()
    simulation.step()

    assert simulation.step_count == 1
    assert simulation.pos.shape == previous_positions.shape
    assert np.all(np.isfinite(simulation.pos))
    assert simulation.pos[0, 0] < previous_positions[0, 0]
    assert simulation.pos[1, 0] > previous_positions[1, 0]
