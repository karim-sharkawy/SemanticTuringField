import string
from typing import Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# These functions are used for user-entered sentence only
# Meaning these functions should appear in gravity_wave.py
def tokenize(sentence: str) -> list[str]:
    return sentence.lower().translate(str.maketrans("", "", string.punctuation)).split()


def stopword_removal(sentence: str) -> list[str]:
    tokens = tokenize(sentence)
    return [word for word in tokens if word not in ENGLISH_STOP_WORDS]


def sentence_to_embedding(
    tokens: list[str], embeddings: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    valid_tokens: list[np.ndarray] = [embeddings[token] for token in tokens if token in embeddings]

    if not valid_tokens:
        return None

    return np.mean(valid_tokens, axis=0)
