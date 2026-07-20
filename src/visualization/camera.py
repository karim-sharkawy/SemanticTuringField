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

import pygame

class Camera:

    def __init__(
        self,
        width=1200,
        height=800,
    ):

        # Window dimensions
        self.width = width
        self.height = height

        # Camera state
        self.zoom = 80.0

        self.min_zoom = 10.0
        self.max_zoom = 400.0

        # World origin appears in the middle
        self.offset = pygame.Vector2(
            width / 2,
            height / 2,
        )

        # Mouse dragging
        self.dragging = False

    ### Coordinate transforms
    def world_to_screen(self, world_pos):
        """
        Convert world coordinates
        to screen pixels.
        """

        return (
            world_pos[0] * self.zoom + self.offset.x,
            world_pos[1] * self.zoom + self.offset.y,
        )

    def screen_to_world(self, screen_pos):
        """
        Convert screen pixels
        to world coordinates.
        """

        return (
            (screen_pos[0] - self.offset.x) / self.zoom,
            (screen_pos[1] - self.offset.y) / self.zoom,
        )

    ### Event handling
    def handle_event(self, event):

        # Begin drag
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
        ):
            self.dragging = True

        # End drag
        elif (
            event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
        ):
            self.dragging = False

        # Drag camera
        elif (
            event.type == pygame.MOUSEMOTION
            and self.dragging
        ):

            self.offset += pygame.Vector2(
                event.rel
            )

        # Mouse wheel zoom
        elif event.type == pygame.MOUSEWHEEL:

            self.zoom_at_mouse(event.y)

    ### Zoom
    def zoom_at_mouse(self, wheel_direction):
        """
        Zoom toward the mouse cursor instead
        of the screen center.
        """

        mouse = pygame.mouse.get_pos()

        # World position BEFORE zoom
        before = self.screen_to_world(mouse)

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
        after = self.screen_to_world(mouse)

        # Adjust offset so the same world
        # point remains beneath the cursor.
        dx = (after[0] - before[0]) * self.zoom
        dy = (after[1] - before[1]) * self.zoom

        self.offset.x += dx
        self.offset.y += dy

    ### Utilities
    def reset(self):
        """
        Reset the camera to its
        default position.
        """

        self.zoom = 80.0

        self.offset = pygame.Vector2(
            self.width / 2,
            self.height / 2,
        )