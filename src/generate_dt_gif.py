"""
Generate a GIF comparing different numerical timesteps (dt)
in the Semantic Turing Field.

Each simulation starts from the same initial configuration
and receives the same sentence. Only dt changes.

Run from the project root:

    python -m src.generate_dt_gif
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

INTRO_FRAMES = 15

RESPONSE_STEPS = 120

OUTRO_FRAMES = 15

OUTPUT_PATH = PROJECT_ROOT / "assets" / "gifs" / "dt_comparison.gif"


# Compare these numerical timesteps.
DT_VALUES = [
    0.005,
    0.010,
    0.025,
]


SENTENCE = (
    "The government and local communities should invest "
    "in renewable energy and adaptation strategies."
)


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


def draw_dt_label(
    renderer: Renderer,
    dt: float,
) -> None:
    """
    Draw a small presentation label showing the current dt.
    """

    font = pygame.font.Font(None, 32)

    text = font.render(
        f"dt = {dt:.3f}",
        True,
        (255, 255, 255),
    )

    renderer.screen.blit(
        text,
        (20, 20),
    )


def main() -> None:
    pygame.init()

    renderer = Renderer(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    camera = Camera(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    camera.zoom = 250.0
    camera.offset = pygame.Vector2(
        WINDOW_WIDTH / 2 - camera.zoom,
        WINDOW_HEIGHT / 2 - camera.zoom,
    )

    input_handler = InputHandler(camera)

    frames = []

    print("Generating dt comparison GIF...")

    # -----------------------------------------------------------------------
    # Run each timestep
    # -----------------------------------------------------------------------

    for dt_index, dt in enumerate(DT_VALUES, start=1):
        print(f"Simulation {dt_index}/{len(DT_VALUES)} with dt={dt:.3f}")

        (
            simulation,
            embeddings,
            words,
            vecs,
            clusters,
            labels,
        ) = build_simulation(
            MAX_WORDS,
            dt=dt,
        )

        input_handler.current_sentence = SENTENCE

        # -------------------------------------------------------------------
        # Initial state
        # -------------------------------------------------------------------

        for _ in range(INTRO_FRAMES):
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

            draw_dt_label(
                renderer,
                dt,
            )

            pygame.display.flip()

            frames.append(surface_to_array(renderer.screen))

        # -------------------------------------------------------------------
        # Simulation
        # -------------------------------------------------------------------

        for step in range(RESPONSE_STEPS):
            simulation.step(
                sentence=SENTENCE,
                embeddings=embeddings,
                vecs=vecs,
            )

            if step % FRAME_INTERVAL == 0:
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

                draw_dt_label(
                    renderer,
                    dt,
                )

                pygame.display.flip()

                frames.append(surface_to_array(renderer.screen))

        # -------------------------------------------------------------------
        # Hold final state
        # -------------------------------------------------------------------

        for _ in range(OUTRO_FRAMES):
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

            draw_dt_label(
                renderer,
                dt,
            )

            pygame.display.flip()

            frames.append(surface_to_array(renderer.screen))

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
