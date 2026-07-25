from pathlib import Path

MAX_WORDS: int = 500

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: str = "data/glove.2024.wikigiga.50d_small.txt"
EMBEDDINGS_PATH: str = "data/embeddings.npy"
DEFAULT_SAVE = PROJECT_ROOT / "data/field_state.json"

ALPHA: float = 0.05
BETA: float = 0.015

DT: float = 0.01
DAMPING: float = 0.995

NUM_CLUSTERS: int = 8

NUM_STEPS: int = 100000
