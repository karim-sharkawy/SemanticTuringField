import string
from typing import Dict, Optional

import numpy as np


def tokenize(sentence: str) -> list[str]:
    return sentence.lower().translate(str.maketrans("", "", string.punctuation)).split()


def sentence_to_embedding(
    tokens: list[str], embeddings: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    valid_tokens: list[np.ndarray] = [embeddings[token] for token in tokens if token in embeddings]

    if not valid_tokens:
        return None

    return np.mean(valid_tokens, axis=0)
