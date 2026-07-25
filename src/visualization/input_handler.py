"""
Input Handler

Processes all keyboard and mouse input for the Semantic
Turing Field application.

Responsibilities
----------------
- Camera controls
- Keyboard shortcuts
- Pause / Resume
- Single-step mode
- Reset simulation
"""

from __future__ import annotations

from typing import Any, Optional

import pygame

from src.utils.save_state import load_state, save_state


class InputHandler:
    def __init__(self, camera: Any) -> None:

        self.camera: Any = camera

        # Simulation state
        self.paused: bool = False
        self.single_step: bool = False

        # Gravity-wave sentence
        self.current_sentence: Optional[str] = None

        # Future feature
        self.gravity_steps_remaining: int = 0

        self.typing = False

        self.text = ""

        self.gravity_frames = 0

        self.current_sentence = None

    def handle_event(
        self,
        event: Any,
        simulation: Any,
    ) -> bool:
        """
        Handle one pygame event.

        Returns
        -------
        bool
            False if application should quit.
        """

        # Quit
        if event.type == pygame.QUIT:
            return False

        # Camera always gets events first.
        self.camera.handle_event(event)

        # Keyboard
        if event.type == pygame.KEYDOWN:
            if self.typing:
                if event.key == pygame.K_RETURN:
                    self.current_sentence = self.text
                    self.gravity_frames = 75
                    self.text = ""
                    self.typing = False

                    return True

                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]

                else:
                    self.text += event.unicode

                return True

            if event.key == pygame.K_RETURN:
                self.typing = True
                return True

            # Quit
            if event.key == pygame.K_ESCAPE:
                return False

            # Pause
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused

            # Single step
            elif event.key == pygame.K_RIGHT:
                self.single_step = True

            # Reset
            elif event.key == pygame.K_r:
                simulation.reset()

            # Alpha
            elif event.key == pygame.K_q:
                simulation.alpha += 0.05

            elif event.key == pygame.K_a:
                simulation.alpha = max(
                    0.0,
                    simulation.alpha - 0.05,
                )

            # Beta
            elif event.key == pygame.K_w:
                simulation.beta += 0.05

            elif event.key == pygame.K_s:
                simulation.beta -= 0.05

            # Damping
            elif event.key == pygame.K_e:
                simulation.damping = min(
                    0.999,
                    simulation.damping + 0.005,
                )

            elif event.key == pygame.K_d:
                simulation.damping = max(
                    0.0,
                    simulation.damping - 0.005,
                )

            # Save
            elif event.key == pygame.K_F5:
                save_state(
                    simulation,
                    self.camera,
                )

            # Load
            elif event.key == pygame.K_F9:
                load_state(
                    simulation,
                    self.camera,
                )

            # Time step
            elif event.key == pygame.K_t:
                simulation.dt += 0.005

            elif event.key == pygame.K_g:
                simulation.dt = max(
                    0.001,
                    simulation.dt - 0.005,
                )

        return True

    def should_step(self) -> bool:
        """
        Determine whether the simulation
        should advance this frame.
        """

        # Normal running
        if not self.paused:
            return True

        # Single-step mode
        if self.single_step:
            self.single_step = False
            return True

        return False

    def use_gravity(self):
        if self.gravity_frames > 0:
            self.gravity_frames -= 1
            return True
        return False
