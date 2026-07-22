from pathlib import Path

MAX_WORDS: int = 500

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: str = "data/glove.2024.wikigiga.50d_small.txt"
EMBEDDINGS_PATH: str = "data/embeddings.npy"

ALPHA: float = 0.15
BETA: float = 0.05

DT: float = 0.05
DAMPING: float = 0.99

NUM_CLUSTERS: int = 8

NUM_STEPS: int = 1000
