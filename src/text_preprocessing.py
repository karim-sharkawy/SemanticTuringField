from typing import List, Dict
import numpy as np
import string

def tokenize(sentence):
    clean_text = sentence.lower()
    clean_text = clean_text.translate(str.maketrans('', '', string.punctuation))
    clean_text = clean_text.split()
    return clean_text

# use case: tokens = tokenize(sentence)

### given that a sentence is input, i'll worry about that later
def sentence_to_embedding(tokens: List[str], embeddings: Dict[str, np.ndarray]):
    valid_tokens = []
    for token in tokens:
        if token in embeddings.keys():
            valid_tokens.append(token)

    if len(valid_tokens) == 0:
        return None
    
    avg_embeddings = [embeddings[token] for token in valid_tokens]
    sentence_embedding = np.mean(avg_embeddings, axis=0)

    return sentence_embedding