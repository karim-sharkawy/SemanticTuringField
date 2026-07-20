"""
Semantic Turing Field
Main Application Loop

Coordinates the simulation, visualization, and user interaction.
"""

import pygame
import numpy as np

from src.nlp.embeddings import load_embeddings

from src.utils.config import *

from src.core.simulate_engine import STFSimulation

from src.nlp.semantics import (
    build_similarity_matrix,
    lower_dimensions,
)

from src.nlp.clustering import cluster_embeddings

from src.visualization.renderer import Renderer
from src.visualization.camera import Camera
from src.visualization.input_handler import InputHandler


def build_simulation():
    """
    Load embeddings and initialize the STF simulation.
    """

    try:
        embeddings = np.load(
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

    words = list(embeddings.keys())

    vecs = np.asarray(
        list(embeddings.values()),
        dtype=np.float64,
    )

    vecs = lower_dimensions(vecs)

    similarity_matrix = build_similarity_matrix(vecs)

    simulation = STFSimulation(
        similarity_matrix,
        alpha=ALPHA,
        beta=BETA,
        dt=DT,
        damping=DAMPING,
    )

    #
    # Static semantic clusters
    #

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


def main():
    """
    Entry point.
    """

    pygame.init()

    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800

    renderer = Renderer(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    camera = Camera(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    input_handler = InputHandler(camera)

    (
        simulation,
        embeddings,
        words,
        vecs,
        clusters,
        labels,
    ) = build_simulation()

    running = True

    while running:

        #
        # Handle Events
        #

        for event in pygame.event.get():

            if not input_handler.handle_event(
                event,
                simulation,
            ):
                running = False
                break

        #
        # Advance Simulation
        #

        if running and input_handler.should_step():

            simulation.step(
                sentence=input_handler.current_sentence,
                embeddings=embeddings,
                vecs=vecs,
            )

        #
        # Draw
        #

        renderer.draw(
            camera=camera,
            positions=simulation.pos,
            clusters=clusters,
            labels=labels,
            words=words,
            simulation=simulation,
            paused=input_handler.paused,
        )

        pygame.display.set_caption(
            f"Semantic Turing Field | "
            f"FPS: {renderer.clock.get_fps():.1f}"
        )

    pygame.quit()


if __name__ == "__main__":
    main()