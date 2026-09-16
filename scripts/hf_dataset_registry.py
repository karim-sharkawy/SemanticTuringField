import os
from datasets import load_dataset
from dotenv import load_dotenv

from download_data import extract_portion

FIELD_SIZES = {
    "500 words": 500,
    "1,000 words": 1000,
    "2,500 words": 2500,
    "5,000 words": 5000,
    "Full vocabulary": None,
}

from src.utils.config import HF_DATATSET

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

def load_hf_dataset():
    dataset = load_dataset(HF_DATATSET)
    return dataset

dataset = load_hf_dataset

def push_to_dataset():
    # Push to the hub using the token
    dataset.push_to_hub(
        repo_id=HF_DATATSET, 
        token=hf_token
    )

for name, size in FIELD_SIZES:
    extract_portion(size, name, )
