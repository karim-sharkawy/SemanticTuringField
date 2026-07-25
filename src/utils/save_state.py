"""
save_state.py

Save and restore the stf.
"""

import json

import numpy as np

from src.utils.config import DEFAULT_SAVE


def save_state(simulation, camera, filename: str = DEFAULT_SAVE):
    data = {
        "positions": simulation.pos.tolist(),
        "velocities": simulation.vel.tolist(),
        "step_count": simulation.step_count,
        "parameters": {
            "alpha": simulation.alpha,
            "beta": simulation.beta,
            "dt": simulation.dt,
            "damping": simulation.damping,
        },
        "camera": {
            "zoom": camera.zoom,
            "offset": [
                camera.offset.x,
                camera.offset.y,
            ],
        },
    }

    with open(filename, "w") as f:
        json.dump(
            data,
            f,
            indent=4,
        )

    print("Saved simulation to '{filename}'")


def load_state(simulation, camera, filename=DEFAULT_SAVE):
    with open(filename, "r") as f:
        data = json.load(f)

    simulation.pos = np.asarray(
        data["positions"],
        dtype=np.float64,
    )

    simulation.vel = np.asarray(
        data["velocities"],
        dtype=np.float64,
    )

    simulation.step_count = data["step_count"]

    simulation.alpha = data["parameters"]["alpha"]
    simulation.beta = data["parameters"]["beta"]
    simulation.dt = data["parameters"]["dt"]
    simulation.damping = data["parameters"]["damping"]

    camera.zoom = data["camera"]["zoom"]

    camera.offset.x = data["camera"]["offset"][0]
    camera.offset.y = data["camera"]["offset"][1]

    print(f"Loaded simulation from '{filename}'")
