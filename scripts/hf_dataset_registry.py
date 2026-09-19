import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.utils.config import HF_DATASET

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

DATA_DIR = Path("data")

FIELD_SIZES = {
    "500 words": 500,
    # "1,000 words": 1000,
    # "2,500 words": 2500,
    # "5,000 words": 5000,
    # "Full vocabulary": None,
}


# Create an authenticated Hugging Face API client
def get_hf_api() -> HfApi:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN was not found in the environment.")

    return HfApi(token=HF_TOKEN)


# Create the Hugging Face dataset repository if it doesn't already exist
def create_dataset_repository() -> None:
    api = get_hf_api()

    api.create_repo(
        repo_id=HF_DATASET,
        repo_type="dataset",
        exist_ok=True,
    )

    print(f"Hugging Face dataset ready: {HF_DATASET}")


# Upload a single file to the Hugging Face dataset repository
def upload_file(
    file_path: str | Path,
    path_in_repo: str,
) -> None:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    api = get_hf_api()

    print(f"Uploading {file_path} → {path_in_repo}")

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=path_in_repo,
        repo_id=HF_DATASET,
        repo_type="dataset",
    )


def upload_raw_glove_files() -> None:
    """
    Upload the full GloVe file and all supported vocabulary
    subsets.
    """

    # okay to keep hardcoded for now, but will change later
    files = {
        "glove.2024.wikigiga.50d_500.txt": "raw/glove.2024.wikigiga.50d_500.txt",
        "glove.2024.wikigiga.50d_1000.txt": "raw/glove.2024.wikigiga.50d_1000.txt",
        "glove.2024.wikigiga.50d_2500.txt": "raw/glove.2024.wikigiga.50d_2500.txt",
        "glove.2024.wikigiga.50d_5000.txt": "raw/glove.2024.wikigiga.50d_5000.txt",
        "glove.2024.wikigiga.50d_full.txt": "raw/glove.2024.wikigiga.50d_full.txt",
    }

    for filename, repo_path in files.items():
        local_path = DATA_DIR / filename

        upload_file(
            local_path,
            repo_path,
        )


def main() -> None:
    create_dataset_repository()

    # upload_raw_glove_files()

    print("All raw GloVe files uploaded.")


if __name__ == "__main__":
    main()
