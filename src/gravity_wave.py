from sklearn.metrics.pairwise import cosine_similarity

from config import DT
from text_preprocessing import sentence_to_embedding, tokenize


def gravity_wave(sentence, embeddings, vecs, pos, vel, strength=0.5):

    tokens = tokenize(sentence)

    sentence_vec = sentence_to_embedding(tokens, embeddings)

    if sentence_vec is None:
        print("No valid tokens present.")
        return vel

    sims = cosine_similarity(sentence_vec.reshape(1, -1), vecs)[0]

    for i in range(len(pos)):
        if sims[i] > 0.3:
            direction = -pos[i]
            vel[i] += direction * strength * sims[i] * DT

    return vel
