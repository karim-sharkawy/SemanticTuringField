"""
Generate a GIF showing how the Semantic Turing Field responds
to different input sentences.

The animation:
    1. Shows the initial field.
    2. Applies a sentence.
    3. Shows the resulting semantic response.
    4. Resets the field.
    5. Repeats for several sentences.

Run from the project root:

    python -m src.generate_sentence_gif
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pygame

from src.app import build_simulation
from src.utils.config import MAX_WORDS
from src.visualization.camera import Camera
from src.visualization.gif_export import (
    save_gif,
    surface_to_array,
)
from src.visualization.input_handler import InputHandler
from src.visualization.renderer import Renderer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

FPS = 15

FRAME_INTERVAL = 3

# Frames showing the field before each sentence is applied.
INTRO_FRAMES = 15

# Number of simulation steps for each sentence.
RESPONSE_STEPS = 150

# Frames showing the final state of each response.
OUTRO_FRAMES = 15

OUTPUT_PATH = PROJECT_ROOT / "assets" / "gifs" / "sentence_response.gif"


SENTENCES = [
    (
        "The government and local communities should invest "
        "in renewable energy and adaptation strategies."
    ),
    ("Machine learning models identify patterns in data and make useful predictions."),
]


def capture_frame(
    renderer: Renderer,
    camera: Camera,
    simulation,
    clusters,
    labels,
    words,
    input_handler: InputHandler,
):
    """
    Draw the current STF state without the interactive controls
    and capture the rendered surface.
    """

    renderer.draw(
        camera=camera,
        positions=simulation.pos,
        clusters=clusters,
        labels=labels,
        words=words,
        simulation=simulation,
        paused=False,
        input_handler=input_handler,
        show_ui=False,
    )

    return surface_to_array(renderer.screen)


def reset_simulation(
    simulation,
    initial_positions,
    initial_velocities,
) -> None:
    """
    Restore the simulation to its original state.
    """

    simulation.pos = initial_positions.copy()
    simulation.vel = initial_velocities.copy()


def main() -> None:
    print("Building STF simulation...")

    (
        simulation,
        embeddings,
        words,
        vecs,
        clusters,
        labels,
    ) = build_simulation(MAX_WORDS)

    # Save the original state so every sentence starts from
    # exactly the same configuration.
    initial_positions = simulation.pos.copy()
    initial_velocities = simulation.vel.copy()

    pygame.init()

    renderer = Renderer(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    camera = Camera(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    # Zoom into the field for presentation.
    camera.zoom = 250.0
    camera.offset = pygame.Vector2(
        WINDOW_WIDTH / 2 - camera.zoom,
        WINDOW_HEIGHT / 2 - camera.zoom,
    )

    input_handler = InputHandler(camera)

    frames = []

    print("Generating sentence-response GIF...")

    # -----------------------------------------------------------------------
    # Process each sentence
    # -----------------------------------------------------------------------

    for sentence_index, sentence in enumerate(SENTENCES, start=1):
        print(f"Sentence {sentence_index}/{len(SENTENCES)}: {sentence}")

        # Reset the field so every sentence gets the same starting point.
        reset_simulation(
            simulation,
            initial_positions,
            initial_velocities,
        )

        input_handler.current_sentence = sentence

        # -------------------------------------------------------------------
        # Initial field
        # -------------------------------------------------------------------

        for _ in range(INTRO_FRAMES):
            frame = capture_frame(
                renderer,
                camera,
                simulation,
                clusters,
                labels,
                words,
                input_handler,
            )

            frames.append(frame)

        # -------------------------------------------------------------------
        # Sentence response
        # -------------------------------------------------------------------

        for step in range(RESPONSE_STEPS):
            simulation.step(
                sentence=sentence,
                embeddings=embeddings,
                vecs=vecs,
            )

            if step % FRAME_INTERVAL == 0:
                frame = capture_frame(
                    renderer,
                    camera,
                    simulation,
                    clusters,
                    labels,
                    words,
                    input_handler,
                )

                frames.append(frame)

        # -------------------------------------------------------------------
        # Hold final state briefly
        # -------------------------------------------------------------------

        for _ in range(OUTRO_FRAMES):
            frame = capture_frame(
                renderer,
                camera,
                simulation,
                clusters,
                labels,
                words,
                input_handler,
            )

            frames.append(frame)

    # -----------------------------------------------------------------------
    # Save GIF
    # -----------------------------------------------------------------------

    print("Saving GIF...")

    save_gif(
        frames,
        OUTPUT_PATH,
        fps=FPS,
    )

    pygame.quit()

    print("Done.")


if __name__ == "__main__":
    main()
