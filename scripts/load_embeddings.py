import numpy as np

from src.config import *


def load_embeddings(path, max_words=MAX_WORDS):
    embeddings = {}  # {"word": np.array([...]), ...}

    with open(path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= max_words:
                break
            split_line = line.strip().split()
            word = split_line[0]
            word_embedding = np.array(split_line[1:], dtype=np.float64)
            embeddings[word] = word_embedding

        return embeddings


if __name__ == "__main__":
    embeddings = load_embeddings(DATA_PATH, max_words=MAX_WORDS)

    np.save(EMBEDDINGS_PATH, embeddings, allow_pickle=True)
