# Utilities for exporting simulations as animated GIFs.
# Captures Pygame surfaces directly and saves them as animated GIFs.

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import pygame


def surface_to_array(surface: pygame.Surface) -> np.ndarray:
    """
    Convert a Pygame surface into an RGB NumPy array.

    Pygame returns arrays in (width, height, channels) format,
    while imageio expects (height, width, channels).
    """

    frame = pygame.surfarray.array3d(surface)

    return np.transpose(frame, (1, 0, 2))


def save_gif(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    fps: int = 20,
) -> None:
    """
    Save RGB frames as an animated GIF.

    Parameters
    ----------
    frames:
        Sequence of RGB NumPy arrays.

    output_path:
        Path where the GIF should be written.

    fps:
        Playback frames per second.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = list(frames)

    if not frames:
        raise ValueError("Cannot create GIF: no frames were provided.")

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    imageio.mimsave(
        output_path,
        frames,
        format="GIF",
        duration=1.0 / fps,
        loop=0,
    )

    print(f"GIF saved: {output_path}")
    print(f"Frames: {len(frames)}")
    print(f"FPS: {fps}")
    print(f"Duration: {len(frames) / fps:.2f} seconds")
