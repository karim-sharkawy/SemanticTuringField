import string
from typing import Dict, List

import numpy as np


def tokenize(sentence: str) -> List[str]:
    return sentence.lower().translate(str.maketrans("", "", string.punctuation)).split()


def sentence_to_embedding(tokens: List[str], embeddings: Dict[str, np.ndarray]):
    valid_tokens = [embeddings[token] for token in tokens if token in embeddings]

    if not valid_tokens:
        return None

    return np.mean(valid_tokens, axis=0)
