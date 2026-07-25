from typing import Dict

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.utils.config import *


def load_embeddings(path: str, max_words: int = MAX_WORDS) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}  # {"word": np.array([...]), ...}

    with open(path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= max_words:
                break
            split_line: list[str] = line.strip().split()
            word: str = split_line[0]
            word_embedding: np.ndarray = np.array(split_line[1:], dtype=np.float64)
            embeddings[word] = word_embedding

        return embeddings

def filter_embeddings(embeddings):

    return {

        word: vector

        for word, vector in embeddings.items()

        if word not in ENGLISH_STOP_WORDS

    }

def save_embeddings(EMBEDDINGS_PATH: str, embeddings: Dict[str, np.ndarray]) -> None:
    np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=True)


if __name__ == "__main__":
    embeddings = load_embeddings(DATA_PATH, max_words=MAX_WORDS)
    embeddings = filter_embeddings(embeddings)

    np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=True)
