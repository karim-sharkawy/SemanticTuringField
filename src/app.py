"""
Semantic Turing Field
Main Application Loop

Coordinates the simulation, visualization, and user interaction.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pygame

from src.core.simulate_engine import STFSimulation
from src.nlp.clustering import cluster_embeddings
from src.nlp.embeddings import load_embeddings
from src.nlp.semantics import build_similarity_matrix, lower_dimensions
from src.utils.config import ALPHA, BETA, DATA_PATH, DAMPING, DT, EMBEDDINGS_PATH, MAX_WORDS, NUM_CLUSTERS, NUM_STEPS
from src.visualization.camera import Camera
from src.visualization.input_handler import InputHandler
from src.visualization.renderer import Renderer


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description="Run the Semantic Turing Field simulation.",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=MAX_WORDS,
        help="Number of GloVe word vectors to load from the corpus.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=NUM_STEPS,
        help="Number of steps to run in headless mode.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Run simulation without the pygame visualization for batch or headless execution.",
    )
    return parser.parse_args(argv)


def build_simulation(max_words: int = MAX_WORDS) -> Tuple[
    STFSimulation, Dict[str, np.ndarray], list[str], np.ndarray, np.ndarray, list[str]
]:
    """
    Load embeddings and initialize the STF simulation.
    """

    embeddings_file = Path(EMBEDDINGS_PATH)
    if max_words != MAX_WORDS:
        embeddings_file = embeddings_file.with_name(
            f"{embeddings_file.stem}_{max_words}{embeddings_file.suffix}"
        )

    try:
        embeddings: Dict[str, np.ndarray] = np.load(
            embeddings_file,
            allow_pickle=True,
        ).item()

    except FileNotFoundError:
        embeddings = load_embeddings(
            DATA_PATH,
            max_words,
        )

        np.save(
            embeddings_file,
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


def main(args: Namespace | list[str] | None = None) -> None:
    """
    Entry point.
    """

    parsed_args = args if isinstance(args, Namespace) else parse_args(args)

    (
        simulation,
        embeddings,
        words,
        vecs,
        clusters,
        labels,
    ) = build_simulation(parsed_args.words)

    if parsed_args.no_viz:
        print(
            f"Running headless simulation with {parsed_args.words} words for {parsed_args.steps} steps."
        )

        for _ in range(parsed_args.steps):
            simulation.step()

        mean_speed = float(np.linalg.norm(simulation.vel, axis=1).mean())
        print(
            f"Headless run complete: {simulation.step_count} steps, mean velocity={mean_speed:.6f}."
        )
        return

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

    running: bool = True

    while running:
        for event in pygame.event.get():
            if not input_handler.handle_event(
                event,
                simulation,
            ):
                running = False
                break

        if input_handler.should_step():
            if input_handler.use_gravity():
                simulation.step(
                    sentence=input_handler.current_sentence,
                    embeddings=embeddings,
                    vecs=vecs,
                )
            else:
                simulation.step()

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

        pygame.display.set_caption(
            f"Semantic Turing Field | FPS: {renderer.clock.get_fps():.1f}"
        )

    pygame.quit()


if __name__ == "__main__":
    main()
