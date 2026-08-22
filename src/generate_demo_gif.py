"""
Generate a GIF demonstrating the Semantic Turing Field.

The animation:
    1. Shows the initial semantic field.
    2. Introduces a sentence.
    3. Applies the sentence gravity wave.
    4. Shows the field evolving over time.

Run from the project root:
    python -m src.generate_demo_gif
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable when this script is run directly.
# ---------------------------------------------------------------------------

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
# GIF configuration
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

FPS = 20

# How many simulation steps occur between captured frames.
FRAME_INTERVAL = 5

# Number of frames showing the initial field before the sentence is applied.
INTRO_FRAMES = 20

# Number of frames during which the gravity wave is active.
GRAVITY_FRAMES = 75

# Number of frames after the gravity wave finishes.
OUTRO_FRAMES = 40

OUTPUT_PATH = PROJECT_ROOT / "assets" / "gifs" / "stf_demo.gif"

SENTENCE = "The government and local communities should invest in renewable energy and adaptation strategies to address climate change."


def capture_frame(
    renderer: Renderer,
    camera: Camera,
    simulation,
    clusters,
    labels,
    words,
    input_handler: InputHandler,
) -> object:
    """
    Draw the current STF state and capture the resulting frame.
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
    )

    return surface_to_array(renderer.screen)


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

    pygame.init()

    renderer = Renderer(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    camera = Camera(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    input_handler = InputHandler(camera)

    # -----------------------------------------------------------------------
    # Set the sentence before rendering.
    #
    # This causes the sentence words to be highlighted in yellow.
    # -----------------------------------------------------------------------

    input_handler.current_sentence = SENTENCE

    frames = []

    print("Generating GIF frames...")

    # -----------------------------------------------------------------------
    # Phase 1: Initial field
    # -----------------------------------------------------------------------

    print("Phase 1/3: Initial field")

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

        camera.zoom = 250.0
        camera.offset = pygame.Vector2(
            WINDOW_WIDTH / 2 - camera.zoom,
            WINDOW_HEIGHT / 2 - camera.zoom,
        )

        frames.append(frame)

    # -----------------------------------------------------------------------
    # Phase 2: Gravity-wave response
    # -----------------------------------------------------------------------

    print("Phase 2/3: Gravity-wave response")

    for frame_index in range(GRAVITY_FRAMES):
        simulation.step(
            sentence=SENTENCE,
            embeddings=embeddings,
            vecs=vecs,
        )

        # Only capture every FRAME_INTERVAL simulation steps.
        if frame_index % FRAME_INTERVAL == 0:
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
    # Phase 3: Continue evolving after the gravity wave
    # -----------------------------------------------------------------------

    print("Phase 3/3: Continued evolution")

    for frame_index in range(OUTRO_FRAMES * FRAME_INTERVAL):
        simulation.step()

        if frame_index % FRAME_INTERVAL == 0:
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
