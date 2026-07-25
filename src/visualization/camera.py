"""
camera.py

Camera system for the Semantic Turing Field.

Responsibilities
----------------
- Convert world coordinates to screen coordinates
- Convert screen coordinates to world coordinates
- Mouse drag panning
- Mouse wheel zooming
"""

from __future__ import annotations

from typing import Any, Tuple

import pygame


class Camera:
    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
    ) -> None:

        # Window dimensions
        self.width: int = width
        self.height: int = height

        # Camera state
        self.zoom: float = 80.0

        self.min_zoom: float = 10.0
        self.max_zoom: float = 400.0

        # World origin appears in the middle
        self.offset: pygame.Vector2 = pygame.Vector2(
            width / 2,
            height / 2,
        )

        # Mouse dragging
        self.dragging: bool = False

    ### Coordinate transforms
    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert world coordinates
        to screen pixels.
        """

        return (
            world_pos[0] * self.zoom + self.offset.x,
            world_pos[1] * self.zoom + self.offset.y,
        )

    def screen_to_world(self, screen_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert screen pixels
        to world coordinates.
        """

        return (
            (screen_pos[0] - self.offset.x) / self.zoom,
            (screen_pos[1] - self.offset.y) / self.zoom,
        )

    ### Event handling
    def handle_event(self, event: Any) -> None:

        # Begin drag
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.dragging = True

        # End drag
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False

        # Drag camera
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.offset += pygame.Vector2(event.rel)

        # Mouse wheel zoom
        elif event.type == pygame.MOUSEWHEEL:
            self.zoom_at_mouse(event.y)

    ### Zoom
    def zoom_at_mouse(self, wheel_direction: int) -> None:
        """
        Zoom toward the mouse cursor instead
        of the screen center.
        """

        mouse: Tuple[int, int] = pygame.mouse.get_pos()

        # World position BEFORE zoom
        before: Tuple[float, float] = self.screen_to_world(mouse)

        # Change zoom
        if wheel_direction > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1

        self.zoom = max(
            self.min_zoom,
            min(self.zoom, self.max_zoom),
        )

        # World position AFTER zoom
        after: Tuple[float, float] = self.screen_to_world(mouse)

        # Adjust offset so the same world
        # point remains beneath the cursor.
        dx: float = (after[0] - before[0]) * self.zoom
        dy: float = (after[1] - before[1]) * self.zoom

        self.offset.x += dx
        self.offset.y += dy

    ### Utilities
    def reset(self) -> None:
        """
        Reset the camera to its
        default position.
        """

        self.zoom = 80.0

        self.offset = pygame.Vector2(
            self.width / 2,
            self.height / 2,
        )
