"""
Precompute Semantic Turing Field configurations for the web application.

This script builds the STF state for each supported vocabulary size:

    500 words
    1,000 words
    2,500 words
    5,000 words
    Full vocabulary

The resulting artifacts are saved under:

    data/precomputed/<size>/

These artifacts can later be uploaded to a Hugging Face Dataset
repository and loaded directly by the Streamlit application.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.nlp.clustering import cluster_embeddings
from src.nlp.embeddings import filter_embeddings, load_embeddings
from src.nlp.semantics import build_similarity_matrix, lower_dimensions
from src.utils.config import (
    ALPHA,
    BETA,
    DAMPING,
    DATA_PATH,
    DT,
    NUM_CLUSTERS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIELD_SIZES: dict[str, int | None] = {
    "500": 500,
    "1000": 1000,
    "2500": 2500,
    "5000": 5000,
    "full": None,
}

OUTPUT_DIR = Path("data/precomputed")

# Fixed seed ensures every precomputed field starts from the same
# deterministic initial state every time the script is run.
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_initial_positions(
    n_words: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Create deterministic initial 2D positions for the STF particles.
    """
    rng = np.random.default_rng(seed)

    return rng.random((n_words, 2)).astype(np.float32) * 2.0


def save_json(path: Path, data: dict) -> None:
    """
    Save metadata as formatted JSON.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_words(path: Path, words: list[str]) -> None:
    """
    Save vocabulary words as JSON.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(words, file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------


def precompute_field(
    size_name: str,
    max_words: int | None,
) -> None:
    """
    Build and save one STF field configuration.
    """

    print("\n" + "=" * 70)
    print(f"PRECOMPUTING FIELD: {size_name}")
    print("=" * 70)

    output_dir = OUTPUT_DIR / size_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load embeddings
    # -----------------------------------------------------------------------

    print("\n[1/7] Loading embeddings...")

    if max_words is None:
        # For the full vocabulary, load the entire source file.
        embeddings = load_embeddings(
            DATA_PATH,
            max_words=10**12,
        )
    else:
        embeddings = load_embeddings(
            DATA_PATH,
            max_words=max_words,
        )

    # Apply the same filtering used by the existing STF pipeline.
    embeddings = filter_embeddings(embeddings)

    words = list(embeddings.keys())

    if not words:
        raise RuntimeError(f"No valid embeddings were loaded for field '{size_name}'.")

    print(f"Using {len(words):,} vocabulary words.")

    # -----------------------------------------------------------------------
    # 2. Convert embeddings to matrix
    # -----------------------------------------------------------------------

    print("\n[2/7] Building embedding matrix...")

    vecs = np.asarray(
        list(embeddings.values()),
        dtype=np.float32,
    )

    print(f"Embedding shape: {vecs.shape}")

    # -----------------------------------------------------------------------
    # 3. PCA / dimensionality reduction
    # -----------------------------------------------------------------------

    print("\n[3/7] Preparing semantic vectors...")

    vecs = lower_dimensions(vecs)

    vecs = np.asarray(
        vecs,
        dtype=np.float32,
    )

    print(f"Semantic vector shape: {vecs.shape}")

    # -----------------------------------------------------------------------
    # 4. Similarity matrix
    # -----------------------------------------------------------------------

    print("\n[4/7] Computing similarity matrix...")

    similarity_matrix = build_similarity_matrix(vecs)

    similarity_matrix = np.asarray(
        similarity_matrix,
        dtype=np.float32,
    )

    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    print(f"Similarity matrix size: {similarity_matrix.nbytes / (1024**2):.2f} MB")

    # -----------------------------------------------------------------------
    # 5. K-Means clustering
    # -----------------------------------------------------------------------

    print("\n[5/7] Computing semantic clusters...")

    clusters, labels = cluster_embeddings(
        vecs,
        words,
        n_clusters=NUM_CLUSTERS,
    )

    clusters = np.asarray(
        clusters,
        dtype=np.int32,
    )

    print(f"Clusters: {NUM_CLUSTERS}")
    print(f"Cluster labels: {labels}")

    # -----------------------------------------------------------------------
    # 6. Initial particle state
    # -----------------------------------------------------------------------

    print("\n[6/7] Creating initial particle state...")

    positions = create_initial_positions(
        n_words=len(words),
    )

    velocities = np.zeros_like(
        positions,
        dtype=np.float32,
    )

    # -----------------------------------------------------------------------
    # 7. Save everything
    # -----------------------------------------------------------------------

    print("\n[7/7] Saving precomputed artifacts...")

    np.save(
        output_dir / "embeddings.npy",
        np.asarray(
            list(embeddings.values()),
            dtype=np.float32,
        ),
    )

    np.save(
        output_dir / "vecs.npy",
        vecs,
    )

    np.save(
        output_dir / "similarity.npy",
        similarity_matrix,
    )

    np.save(
        output_dir / "clusters.npy",
        clusters,
    )

    np.save(
        output_dir / "positions.npy",
        positions,
    )

    np.save(
        output_dir / "velocities.npy",
        velocities,
    )

    save_words(
        output_dir / "words.json",
        words,
    )

    save_json(
        output_dir / "labels.json",
        {
            "labels": labels,
        },
    )

    metadata = {
        "field_name": size_name,
        "requested_words": max_words,
        "actual_words": len(words),
        "embedding_dimensions": int(vecs.shape[1]),
        "num_clusters": NUM_CLUSTERS,
        "random_seed": RANDOM_SEED,
        "alpha": ALPHA,
        "beta": BETA,
        "dt": DT,
        "damping": DAMPING,
        "source_data": str(DATA_PATH),
        "artifacts": [
            "embeddings.npy",
            "vecs.npy",
            "similarity.npy",
            "clusters.npy",
            "positions.npy",
            "velocities.npy",
            "words.json",
            "labels.json",
            "metadata.json",
        ],
    }

    save_json(
        output_dir / "metadata.json",
        metadata,
    )

    print("\nSaved:")
    for path in sorted(output_dir.iterdir()):
        size_mb = path.stat().st_size / (1024**2)
        print(f"  {path.name:<20} {size_mb:>8.2f} MB")

    print(f"\nCompleted field: {size_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Precompute all supported STF field sizes.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("SEMANTIC TURING FIELD — PRECOMPUTATION")
    print("=" * 70)

    print("\nFields to precompute:")

    for name, size in FIELD_SIZES.items():
        description = "full vocabulary" if size is None else f"{size:,} words"
        print(f"  {name:<8} → {description}")

    print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")

    for size_name, max_words in FIELD_SIZES.items():
        precompute_field(
            size_name=size_name,
            max_words=max_words,
        )

    print("\n" + "=" * 70)
    print("ALL STF FIELDS PRECOMPUTED")
    print("=" * 70)


if __name__ == "__main__":
    main()
