# Utilities for preprocessing user-entered text and defining vocabulary filtering rules.

import string
from typing import Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def is_valid_vocabulary_word(word: str) -> bool:
    """
    Determine whether a word should be included
    in the STF vocabulary.

    A valid word must:
    - not be empty
    - not be a stopword
    - not consist entirely of punctuation
    """

    if not word:
        return False

    if word in ENGLISH_STOP_WORDS:
        return False

    if len(word) < 4:
        return False

    if all(character in string.punctuation for character in word):
        return False  # removes punctuation only strings, not strings containing punctuation

    return True


def clean_text(sentence: str) -> str:
    sentence = sentence.lower()  # lowercase

    sentence = sentence.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )  # remove punctuation

    return sentence


def tokenize(sentence: str) -> list[str]:
    return clean_text(sentence).split()


def stopword_removal(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in ENGLISH_STOP_WORDS]


def sentence_to_embedding(
    tokens: list[str], embeddings: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    # Compute the mean embedding of all valid tokens in the embedding vocabulary
    valid_tokens: list[np.ndarray] = [embeddings[token] for token in tokens if token in embeddings]

    if len(valid_tokens) == 0:
        return None

    return np.mean(valid_tokens, axis=0)
