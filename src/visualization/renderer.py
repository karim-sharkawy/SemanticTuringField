"""
Renderer

Responsible for drawing the Semantic Turing Field.

Responsibilities
----------------
- Draw particles
- Draw hover labels
- Draw UI text
- Maintain 60 FPS
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pygame

from src.nlp.text_preprocessing import stopword_removal, tokenize
from src.visualization.colors import cluster_color


class Renderer:
    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
    ) -> None:
        pygame.init()

        self.width: int = width
        self.height: int = height

        self.screen: pygame.Surface = pygame.display.set_mode((width, height))

        pygame.display.set_caption("Semantic Turing Field")

        self.clock: pygame.time.Clock = pygame.time.Clock()

        self.font: pygame.font.Font = pygame.font.SysFont(
            "Arial",
            16,
        )

        self.background: tuple[int, int, int] = (
            18,
            18,
            18,
        )

        # Normal particle size
        self.particle_radius: int = 4

        # Highlighted sentence particle size
        self.sentence_particle_radius: int = 7

        # Highlight color for words appearing in
        # the user's sentence.
        self.sentence_color: tuple[int, int, int] = (
            255,
            220,
            80,
        )

    def draw(
        self,
        camera: Any,
        positions: np.ndarray,
        clusters: np.ndarray,
        labels: list[str],
        words: list[str],
        simulation: Any,
        paused: bool,
        input_handler: Any = None,
        show_ui: bool = True,
    ) -> None:
        self.screen.fill(self.background)

        # Determine which particles belong to the current sentence.
        if input_handler is not None:
            sentence = input_handler.current_sentence
        else:
            sentence = None

        sentence_indices = self.get_sentence_indices(
            sentence,
            words,
        )

        self.draw_particles(
            camera,
            positions,
            clusters,
            sentence_indices,
        )

        self.draw_hover_label(
            camera,
            positions,
            words,
        )

        if show_ui:
            self.draw_ui(
                simulation,
                paused,
            )

            if input_handler is not None:
                self.draw_sentence_box(input_handler)

        pygame.display.flip()
        self.clock.tick(60)

    # Sentence highlighting
    def get_sentence_indices(
        self,
        sentence: str | None,
        words: list[str],
    ) -> set[int]:
        """
        Return the particle indices corresponding
        to valid words in the user's current sentence.

        Stopwords and words outside the vocabulary
        are ignored.
        """

        if not sentence:
            return set()

        tokens = tokenize(sentence)

        tokens = stopword_removal(tokens)

        sentence_words = set(tokens)

        return {index for index, word in enumerate(words) if word in sentence_words}

    # Particle Drawing
    def draw_particles(
        self,
        camera: Any,
        positions: np.ndarray,
        clusters: np.ndarray,
        sentence_indices: set[int],
    ) -> None:
        for i, pos in enumerate(positions):
            x, y = camera.world_to_screen(pos)

            if not np.isfinite(x) or not np.isfinite(y):
                print(f"Invalid particle {i}: ({x}, {y})")
                print(f"Raw position: {pos}")
                raise RuntimeError("Particle position became invalid.")

            x = int(round(float(x)))
            y = int(round(float(y)))

            if i in sentence_indices:
                color = self.sentence_color
                radius = self.sentence_particle_radius
            else:
                color = cluster_color(clusters[i])
                radius = self.particle_radius

            pygame.draw.circle(
                self.screen,
                color,
                (x, y),
                radius,
            )

    # Hover Labels
    def draw_hover_label(
        self,
        camera: Any,
        positions: np.ndarray,
        words: list[str],
    ) -> None:
        mouse: tuple[int, int] = pygame.mouse.get_pos()

        hovered: int | None = None

        for i, pos in enumerate(positions):
            sx, sy = camera.world_to_screen(pos)

            dx = mouse[0] - sx
            dy = mouse[1] - sy

            if dx * dx + dy * dy < 100:
                hovered = i
                break

        if hovered is None:
            return

        label: pygame.Surface = self.font.render(
            words[hovered],
            True,
            (255, 255, 255),
        )

        self.screen.blit(
            label,
            (
                mouse[0] + 12,
                mouse[1] + 12,
            ),
        )

    # UI
    def draw_ui(
        self,
        simulation: Any,
        paused: bool,
    ) -> None:
        status = "Paused" if paused else "Running"

        ui: list[str] = [
            f"Status: {status}",
            f"Alpha: {simulation.alpha:.2f}",
            f"Beta: {simulation.beta:.2f}",
            f"Damping: {simulation.damping:.3f}",
            f"dt: {simulation.dt:.3f}",
            "",
            "Q/A : Alpha",
            "W/S : Beta",
            "E/D : Damping",
            "T/G : dt",
            "R : Reset",
            "SPACE : Pause",
            "RIGHT : Step",
            "Mouse Wheel : Zoom",
            "Drag : Pan",
            "F5 : Save State",
            "F9 : Load State",
            "",
            "Yellow particles = sentence words",
        ]

        y: int = 10

        for line in ui:
            surface: pygame.Surface = self.font.render(
                line,
                True,
                (230, 230, 230),
            )

            self.screen.blit(
                surface,
                (
                    10,
                    y,
                ),
            )

            y += 22

    # Sentence Input Box
    def draw_sentence_box(
        self,
        input_handler: Any,
    ) -> None:
        y = self.height - 45

        pygame.draw.rect(
            self.screen,
            (40, 40, 40),
            (
                0,
                y,
                self.width,
                45,
            ),
        )

        if input_handler.typing:
            text = "> " + input_handler.text + "_"

        elif input_handler.current_sentence:
            text = "Sentence: " + input_handler.current_sentence

        else:
            text = "Press ENTER to type a sentence"

        surface = self.font.render(
            text,
            True,
            (255, 255, 255),
        )

        self.screen.blit(
            surface,
            (
                10,
                y + 12,
            ),
        )
