MAX_WORDS = 500

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = "data/glove.2024.wikigiga.50d_small.txt"
EMBEDDINGS_PATH = "data/embeddings.npy"

ALPHA = 0.95
BETA = 0.2

DT = 0.05
DAMPING = 0.99

NUM_CLUSTERS = 8

NUM_STEPS = 1000
