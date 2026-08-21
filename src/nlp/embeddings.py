# Loading and preprocessing of word embeddings used by the STF

from typing import Dict

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.nlp.text_preprocessing import is_valid_vocabulary_word
from src.utils.config import *


def load_embeddings(path: str, max_words: int = MAX_WORDS) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if len(embeddings) >= max_words:  # Stop once we have enough valid words
                break  # instead of index >= max_words

            parts = line.strip().split()

            if len(parts) < 2:
                continue

            word = parts[0]

            if not is_valid_vocabulary_word(word):
                continue  # Ignore unwanted vocabulary entries

            try:
                vector = np.asarray(
                    parts[1:],
                    dtype=np.float64,
                )

            except ValueError:
                continue

            embeddings[word] = vector

    print(f"Loaded {len(embeddings)} valid embeddings.")

    return embeddings


def filter_embeddings(embeddings):
    return {word: vector for word, vector in embeddings.items() if word not in ENGLISH_STOP_WORDS}


def save_embeddings(EMBEDDINGS_PATH: str, embeddings: Dict[str, np.ndarray]) -> None:
    np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=True)


if __name__ == "__main__":
    embeddings = load_embeddings(DATA_PATH, max_words=MAX_WORDS)
    embeddings = filter_embeddings(embeddings)

    np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=True)
