# should break this up into two config files
from pathlib import Path

MAX_WORDS: int = 500

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_PATH: str = "data/glove.2024.wikigiga.50d_small.txt"
FULL_DATA : str = "data/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"

EMBEDDINGS_PATH: str = f"data/embeddings_{MAX_WORDS}.npy"
DEFAULT_SAVE = PROJECT_ROOT / "data/field_state.json"

HF_DATASET = "KrispyKarim/STF_Embeddings"

ALPHA: float = 0.05
BETA: float = 0.015

DT: float = 0.01
DAMPING: float = 0.995

NUM_CLUSTERS: int = 8

NUM_STEPS: int = 5000
