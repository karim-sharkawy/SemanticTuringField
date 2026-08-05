from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.app import build_simulation


def initialize_simulation(word_limit: int):
    return build_simulation(word_limit)


def render_field(simulation, clusters, words, steps, sentence, gravity_frames, embeddings, vecs):
    for _ in range(steps):
        if gravity_frames > 0 and sentence:
            simulation.step(sentence=sentence, embeddings=embeddings, vecs=vecs)
            gravity_frames -= 1
        else:
            simulation.step()

    fig, ax = plt.subplots(figsize=(8, 6))
    positions = simulation.pos

    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=clusters,
        cmap="tab10",
        s=30,
        alpha=0.75,
        edgecolors="none",
    )

    ax.set_title("Semantic Turing Field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    return fig, gravity_frames


def main():
    st.set_page_config(page_title="Semantic Turing Field Web UI", layout="wide")
    st.title("Semantic Turing Field — Interactive Browser Demo")

    sidebar = st.sidebar
    sidebar.header("Simulation Settings")

    words = sidebar.slider(
        "Word vectors",
        min_value=500,
        max_value=10000,
        step=500,
        value=1000,
    )
    steps = sidebar.slider(
        "Steps per update",
        min_value=1,
        max_value=100,
        step=1,
        value=25,
    )

    sentence = sidebar.text_input(
        "Sentence",
        value=st.session_state.get("sentence", ""),
        help="Type a sentence and click Apply to perturb the semantic field.",
    )

    if "simulation" not in st.session_state or st.session_state.get("words") != words:
        simulation, embeddings, words_list, vecs, clusters, labels = initialize_simulation(words)
        st.session_state.simulation = simulation
        st.session_state.embeddings = embeddings
        st.session_state.words_list = words_list
        st.session_state.vecs = vecs
        st.session_state.clusters = clusters
        st.session_state.labels = labels
        st.session_state.words = words
        st.session_state.gravity_frames = 0
        st.session_state.current_sentence = ""

    if sidebar.button("Reset field"):
        st.session_state.simulation.reset()

    if sidebar.button("Apply sentence"):
        st.session_state.current_sentence = sentence
        st.session_state.gravity_frames = 50

    if st.session_state.current_sentence:
        st.markdown(f"**Current sentence:** {st.session_state.current_sentence}")

    fig, new_gravity_frames = render_field(
        st.session_state.simulation,
        st.session_state.clusters,
        st.session_state.words_list,
        steps,
        st.session_state.current_sentence,
        st.session_state.gravity_frames,
        st.session_state.embeddings,
        st.session_state.vecs,
    )

    st.session_state.gravity_frames = new_gravity_frames

    st.pyplot(fig)

    stats_col, _, controls_col = st.columns([3, 1, 2])
    with stats_col:
        st.metric("Vocabulary size", str(words))
        st.metric("Simulation steps", str(st.session_state.simulation.step_count))
        st.metric("Gravity frames remaining", str(st.session_state.gravity_frames))

    with controls_col:
        st.write("### Controls")
        st.write("- Enter sentence in sidebar")
        st.write("- Apply sentence to perturb the field")
        st.write("- Reset field to random positions")
        st.write("- Change vocabulary size to reload embeddings")


if __name__ == "__main__":
    main()
