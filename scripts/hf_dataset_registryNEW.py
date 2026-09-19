from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from src.core.simulate_engine import STFSimulation
from src.utils.config import (
    ALPHA,
    BETA,
    DAMPING,
    DT,
    HF_DATASET,
    NUM_CLUSTERS,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

FIELD_SIZES = {
    "500 words": 500,
    "1,000 words": 1000,
    "2,500 words": 2500,
    "5,000 words": 5000,
    "Full vocabulary": None,
}


# ---------------------------------------------------------------------
# Hugging Face helpers
# ---------------------------------------------------------------------


def get_hf_api() -> HfApi:
    """
    Return an authenticated Hugging Face API client.

    HF_TOKEN is optional for public dataset repositories, but required
    if the repository is private.
    """

    if HF_TOKEN:
        return HfApi(token=HF_TOKEN)

    return HfApi()


def download_field_file(
    field_name: str,
    filename: str,
) -> Path:
    """
    Download one artifact for a precomputed STF field from Hugging Face.

    The Hugging Face Hub handles local caching automatically, so files
    are only downloaded again when necessary.
    """

    if field_name not in FIELD_SIZES:
        raise ValueError(
            f"Unknown field '{field_name}'. "
            f"Available fields: {list(FIELD_SIZES.keys())}"
        )

    size = FIELD_SIZES[field_name]

    directory_name = "full" if size is None else str(size)

    repo_path = f"precomputed/{directory_name}/{filename}"

    downloaded_path = hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename=repo_path,
        token=HF_TOKEN,
    )

    return Path(downloaded_path)


# ---------------------------------------------------------------------
# Load precomputed field
# ---------------------------------------------------------------------


def load_precomputed_field(field_name: str):
    """
    Load a complete precomputed STF configuration from Hugging Face.

    Returns:
        simulation
        embeddings
        words
        vecs
        clusters
        labels
    """

    if field_name not in FIELD_SIZES:
        raise ValueError(
            f"Unknown field '{field_name}'. "
            f"Available fields: {list(FIELD_SIZES.keys())}"
        )

    print(f"Loading '{field_name}' from Hugging Face...")

    # ---------------------------------------------------------------
    # Download artifacts
    # ---------------------------------------------------------------

    embeddings_path = download_field_file(
        field_name,
        "embeddings.npy",
    )

    vecs_path = download_field_file(
        field_name,
        "vecs.npy",
    )

    similarity_path = download_field_file(
        field_name,
        "similarity.npy",
    )

    clusters_path = download_field_file(
        field_name,
        "clusters.npy",
    )

    positions_path = download_field_file(
        field_name,
        "positions.npy",
    )

    velocities_path = download_field_file(
        field_name,
        "velocities.npy",
    )

    words_path = download_field_file(
        field_name,
        "words.json",
    )

    labels_path = download_field_file(
        field_name,
        "labels.json",
    )

    # ---------------------------------------------------------------
    # Load NumPy artifacts
    # ---------------------------------------------------------------

    embeddings_array = np.load(embeddings_path)

    vecs = np.load(vecs_path)

    similarity_matrix = np.load(
        similarity_path,
        mmap_mode="r",
    )

    clusters = np.load(clusters_path)

    positions = np.load(positions_path)

    velocities = np.load(velocities_path)

    # ---------------------------------------------------------------
    # Load words
    # ---------------------------------------------------------------

    with words_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        words = json.load(file)

    # ---------------------------------------------------------------
    # Load cluster labels
    # ---------------------------------------------------------------

    with labels_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        labels_data = json.load(file)

    labels = labels_data["labels"]

    # ---------------------------------------------------------------
    # Reconstruct embedding dictionary
    # ---------------------------------------------------------------

    embeddings = {
        word: embedding
        for word, embedding in zip(
            words,
            embeddings_array,
        )
    }

    # ---------------------------------------------------------------
    # Reconstruct simulation
    # ---------------------------------------------------------------

    simulation = STFSimulation(
        similarity_matrix=similarity_matrix,
        alpha=ALPHA,
        beta=BETA,
        dt=DT,
        damping=DAMPING,
    )

    # Replace random initialization with the canonical precomputed
    # starting state.
    simulation.pos = np.array(
        positions,
        copy=True,
    )

    simulation.vel = np.array(
        velocities,
        copy=True,
    )

    simulation.step_count = 0

    print(
        f"Loaded {len(words):,} words "
        f"from Hugging Face."
    )

    return (
        simulation,
        embeddings,
        words,
        vecs,
        clusters,
        labels,
    )


# ---------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------


def create_dataset_repository():
    """
    Create the Hugging Face Dataset repository if it does not exist.
    """

    api = get_hf_api()

    api.create_repo(
        repo_id=HF_DATASET,
        repo_type="dataset",
        exist_ok=True,
    )

    print(
        f"Hugging Face dataset ready: {HF_DATASET}"
    )


def upload_file(
    file_path: str | Path,
    path_in_repo: str,
):
    """
    Upload one local file to the Hugging Face Dataset repository.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    api = get_hf_api()

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=path_in_repo,
        repo_id=HF_DATASET,
        repo_type="dataset",
    )

    print(
        f"Uploaded {file_path} → {path_in_repo}"
    )


def upload_precomputed_fields():
    """
    Upload all locally precomputed STF fields to Hugging Face.
    """

    project_root = Path(__file__).resolve().parents[2]

    precomputed_dir = (
        project_root
        / "data"
        / "precomputed"
    )

    create_dataset_repository()

    for field_name, size in FIELD_SIZES.items():

        directory_name = (
            "full"
            if size is None
            else str(size)
        )

        field_dir = (
            precomputed_dir
            / directory_name
        )

        if not field_dir.exists():
            raise FileNotFoundError(
                f"Precomputed field not found: {field_dir}"
            )

        print(
            f"\nUploading field: {field_name}"
        )

        for file_path in field_dir.iterdir():

            if not file_path.is_file():
                continue

            repo_path = (
                f"fields/"
                f"{directory_name}/"
                f"{file_path.name}"
            )

            upload_file(
                file_path,
                repo_path,
            )

    print(
        "\nAll precomputed STF fields uploaded."
    )


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------


if __name__ == "__main__":
    upload_precomputed_fields()