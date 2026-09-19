from __future__ import annotations

import os
import sys
from pathlib import Path

import pygame
import streamlit as st

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.hf_dataset_registryNEW import (
    FIELD_SIZES,
    load_precomputed_field,
)
from src.nlp.text_preprocessing import tokenize
from src.visualization.camera import Camera
from src.visualization.renderer import Renderer

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_FIELD = "1,000 words"

GRAVITY_FRAMES = 50


# ---------------------------------------------------------------------
# Cached simulation initialization
# ---------------------------------------------------------------------


@st.cache_resource
def initialize_simulation(field_name: str):
    """
    Load a precomputed STF configuration from Hugging Face.

    Streamlit caches the loaded configuration so the Hugging Face
    artifacts are not repeatedly downloaded during reruns.
    """

    return load_precomputed_field(field_name)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------


def initialize_session_state():
    """Create Streamlit session state variables."""

    if "field_size" not in st.session_state:
        st.session_state.field_size = DEFAULT_FIELD

    if "simulation" not in st.session_state:
        st.session_state.simulation = None

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None

    if "words_list" not in st.session_state:
        st.session_state.words_list = None

    if "vecs" not in st.session_state:
        st.session_state.vecs = None

    if "clusters" not in st.session_state:
        st.session_state.clusters = None

    if "labels" not in st.session_state:
        st.session_state.labels = None

    if "gravity_frames" not in st.session_state:
        st.session_state.gravity_frames = 0

    if "current_sentence" not in st.session_state:
        st.session_state.current_sentence = ""

    if "running" not in st.session_state:
        st.session_state.running = False


# ---------------------------------------------------------------------
# Field loading
# ---------------------------------------------------------------------


def load_field(field_name: str):
    """
    Load one of the five supported STF configurations from Hugging Face.
    """

    (
        simulation,
        embeddings,
        words,
        vecs,
        clusters,
        labels,
    ) = initialize_simulation(field_name)

    st.session_state.simulation = simulation
    st.session_state.embeddings = embeddings
    st.session_state.words_list = words
    st.session_state.vecs = vecs
    st.session_state.clusters = clusters
    st.session_state.labels = labels

    st.session_state.gravity_frames = 0
    st.session_state.current_sentence = ""
    st.session_state.running = False


# ---------------------------------------------------------------------
# Pygame rendering
# ---------------------------------------------------------------------


def render_field():
    """
    Render the current STF state using the same Renderer used by
    the local Pygame application.
    """

    simulation = st.session_state.simulation

    width = 1200
    height = 800

    # Create renderer once.
    if st.session_state.get("renderer") is None:
        st.session_state.renderer = Renderer(
            width=width,
            height=height,
        )

    renderer = st.session_state.renderer

    # Create camera once.
    if st.session_state.get("camera") is None:
        camera = Camera(
            width=width,
            height=height,
        )

        camera.zoom = 250.0
        camera.offset = pygame.Vector2(
            width / 2 - camera.zoom,
            height / 2 - camera.zoom,
        )

        st.session_state.camera = camera

    camera = st.session_state.camera

    renderer.draw(
        camera=camera,
        positions=simulation.pos,
        clusters=st.session_state.clusters,
        labels=st.session_state.labels,
        words=st.session_state.words_list,
        simulation=simulation,
        paused=not st.session_state.running,
        input_handler=None,
        show_ui=False,
    )

    # Convert the Pygame surface to an image Streamlit can display.
    image = pygame.surfarray.array3d(renderer.screen)
    image = image.swapaxes(0, 1)

    return image


# ---------------------------------------------------------------------
# Sentence handling
# ---------------------------------------------------------------------


def get_sentence_words(sentence: str):
    """
    Return vocabulary words from the sentence that exist in the
    currently loaded STF field.
    """

    if not sentence:
        return []

    tokens = tokenize(sentence)

    embeddings = st.session_state.embeddings

    if embeddings is None:
        return []

    return [token for token in tokens if token in embeddings]


def apply_sentence():
    """
    Apply the current sentence as a gravity-wave disturbance.
    """

    sentence = st.session_state.sentence_input.strip()

    if not sentence:
        st.warning("Enter a sentence first.")
        return

    # Applying a sentence automatically runs the simulation too
    st.session_state.current_sentence = sentence
    st.session_state.gravity_frames = GRAVITY_FRAMES
    st.session_state.running = True


# ---------------------------------------------------------------------
# Simulation update
# ---------------------------------------------------------------------


def update_simulation():
    """
    Advance the STF simulation.
    """

    simulation = st.session_state.simulation

    if simulation is None:
        return

    sentence = st.session_state.current_sentence

    if st.session_state.gravity_frames > 0 and sentence:
        simulation.step(
            sentence=sentence,
            embeddings=st.session_state.embeddings,
            vecs=st.session_state.vecs,
        )

        st.session_state.gravity_frames -= 1

    else:
        simulation.step()


@st.fragment(run_every=0.1)
def simulation_view():
    """
    Continuously update and render the STF simulation.

    The fragment reruns independently of the rest of the Streamlit app,
    allowing the field to animate without requiring user interaction.
    """

    if st.session_state.simulation is None:
        return

    if st.session_state.running:
        update_simulation()

    image = render_field()

    st.image(
        image,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Semantic Turing Field",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------

    st.title("Semantic Turing Field")

    st.markdown(
        """
        A universe of words that evolves toward meaning.

        Explore how word embeddings interact as a dynamic semantic field,
        then disturb the field with a sentence and watch it evolve.
        """
    )

    st.divider()

    # -----------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------

    with st.sidebar:
        st.header("Field")

        selected_field = st.radio(
            "Vocabulary size",
            options=list(FIELD_SIZES.keys()),
            index=list(FIELD_SIZES.keys()).index(st.session_state.field_size),
        )

        if selected_field != st.session_state.field_size:
            st.session_state.field_size = selected_field

            # Clear cached visualization objects.
            st.session_state.renderer = None
            st.session_state.camera = None

            load_field(selected_field)

            st.rerun()

        st.divider()

        st.header("Simulation")

        if st.button(
            "▶ Run",
            use_container_width=True,
        ):
            st.session_state.running = True

        if st.button(
            "Ⅱ Pause",
            use_container_width=True,
        ):
            st.session_state.running = False

        if st.button(
            "↻ Reset",
            use_container_width=True,
        ):
            st.session_state.simulation.reset()
            st.session_state.gravity_frames = 0
            st.session_state.current_sentence = ""
            st.session_state.running = False

        st.divider()

        st.caption(
            "Choose a field size above. Each configuration is "
            "precomputed and optimized for the web demo."
        )

    # -----------------------------------------------------------------
    # Load initial field
    # -----------------------------------------------------------------

    if st.session_state.simulation is None:
        with st.spinner(f"Loading {st.session_state.field_size}..."):
            load_field(st.session_state.field_size)

    # -----------------------------------------------------------------
    # Main field
    # -----------------------------------------------------------------

    field_col, info_col = st.columns(
        [4, 1],
        gap="large",
    )

    with field_col:
        st.subheader("Semantic Field")

        simulation_view()

    # -----------------------------------------------------------------
    # Information panel
    # -----------------------------------------------------------------

    with info_col:
        st.subheader("Field Info")

        vocabulary_size = len(st.session_state.words_list)

        st.metric(
            "Vocabulary",
            f"{vocabulary_size:,}",
        )

        st.metric(
            "Simulation steps",
            f"{st.session_state.simulation.step_count:,}",
        )

        if st.session_state.gravity_frames > 0:
            st.metric(
                "Gravity frames",
                str(st.session_state.gravity_frames),
            )
        else:
            st.metric(
                "Gravity frames",
                "Inactive",
            )

        st.divider()

        st.caption(f"Field: {st.session_state.field_size}")

        st.caption(
            "The field evolves according to semantic "
            "relationships between the loaded word embeddings."
        )

    # -----------------------------------------------------------------
    # Sentence interaction
    # -----------------------------------------------------------------

    st.divider()

    st.subheader("Disturb the Field")

    st.write("Enter a sentence to create a semantic gravity wave.")

    sentence_col, button_col = st.columns(
        [5, 1],
        vertical_alignment="bottom",
    )

    with sentence_col:
        st.text_input(
            "Sentence",
            key="sentence_input",
            placeholder=("e.g. Machine learning models identify patterns in data."),
            label_visibility="collapsed",
        )

    with button_col:
        st.button(
            "Apply sentence",
            use_container_width=True,
            on_click=apply_sentence,
        )

    # -----------------------------------------------------------------
    # Sentence information
    # -----------------------------------------------------------------

    if st.session_state.current_sentence:
        vocabulary_words = get_sentence_words(st.session_state.current_sentence)

        st.markdown(f"Current sentence: {st.session_state.current_sentence}")

        if vocabulary_words:
            st.caption(
                f"{len(vocabulary_words)} vocabulary words "
                f"found in the field: "
                f"{', '.join(vocabulary_words)}"
            )

        else:
            st.caption("No words from this sentence were found in the current vocabulary.")

    # -----------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------

    with st.expander("How the Semantic Turing Field works"):
        st.markdown(
            """
            The Semantic Turing Field represents words as particles
            in a dynamic two-dimensional space.

            Words with stronger semantic relationships influence one
            another through the simulation's force model. Over time,
            the field evolves into a structure reflecting relationships
            encoded in the underlying word embeddings.

            A sentence can be introduced as a temporary disturbance.
            Its embedding interacts with the field and creates a
            semantic gravity wave that changes the motion of
            nearby words.
            """
        )


if __name__ == "__main__":
    main()
