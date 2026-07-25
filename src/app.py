"""
Semantic Turing Field
Main Application Loop

Coordinates the simulation, visualization, and user interaction.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import pygame

from src.core.simulate_engine import STFSimulation
from src.nlp.clustering import cluster_embeddings
from src.nlp.embeddings import load_embeddings
from src.nlp.semantics import build_similarity_matrix, lower_dimensions
from src.utils.config import *
from src.visualization.camera import Camera
from src.visualization.input_handler import InputHandler
from src.visualization.renderer import Renderer


def build_simulation() -> Tuple[
    STFSimulation, Dict[str, np.ndarray], list[str], np.ndarray, np.ndarray, list[str]
]:
    """
    Load embeddings and initialize the STF simulation.
    """

    try:
        embeddings: Dict[str, np.ndarray] = np.load(
            EMBEDDINGS_PATH,
            allow_pickle=True,
        ).item()

    except FileNotFoundError:
        embeddings = load_embeddings(
            DATA_PATH,
            MAX_WORDS,
        )

        np.save(
            EMBEDDINGS_PATH,
            embeddings,
            allow_pickle=True,
        )

    words: list[str] = list(embeddings.keys())

    vecs: np.ndarray = np.asarray(
        list(embeddings.values()),
        dtype=np.float64,
    )

    vecs = lower_dimensions(vecs)

    similarity_matrix: np.ndarray = build_similarity_matrix(vecs)

    simulation: STFSimulation = STFSimulation(
        similarity_matrix,
        alpha=ALPHA,
        beta=BETA,
        dt=DT,
        damping=DAMPING,
    )

    #
    # Static semantic clusters
    #

    clusters: np.ndarray
    labels: list[str]
    clusters, labels = cluster_embeddings(
        vecs,
        words,
        NUM_CLUSTERS,
    )

    return (
        simulation,
        embeddings,
        words,
        vecs,
        clusters,
        labels,
    )


def main() -> None:
    """
    Entry point.
    """

    pygame.init()

    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 800

    renderer: Renderer = Renderer(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    camera: Camera = Camera(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    input_handler: InputHandler = InputHandler(camera)

    (
        simulation,
        embeddings,
        words,
        vecs,
        clusters,
        labels,
    ) = build_simulation()

    running: bool = True

    while running:
        # Handle Events
        for event in pygame.event.get():
            if not input_handler.handle_event(
                event,
                simulation,
            ):
                running = False
                break

        # Advance Simulation
        if input_handler.use_gravity():
            simulation.step(
                sentence=input_handler.current_sentence,
                embeddings=embeddings,
                vecs=vecs,
            )

        else: simulation.step()

        # Draw
        renderer.draw(
            camera=camera,
            positions=simulation.pos,
            clusters=clusters,
            labels=labels,
            words=words,
            simulation=simulation,
            paused=input_handler.paused,
            input_handler=input_handler,
        )

        pygame.display.set_caption(f"Semantic Turing Field | FPS: {renderer.clock.get_fps():.1f}")

    pygame.quit()


if __name__ == "__main__":
    print("Starting timer...")
    start_time: float = time.time()

    try:
        main()
    finally:
        end_time: float = time.time()
        elapsed_time: float = end_time - start_time

        print(f"Code exucted in: {elapsed_time:.4f} seconds")
