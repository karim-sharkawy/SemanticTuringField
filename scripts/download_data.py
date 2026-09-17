import os
import urllib.request
import zipfile
from pathlib import Path
from functools import lru_cache


GLOVE_URL = (
    "https://nlp.stanford.edu/data/wordvecs/"
    "glove.2024.wikigiga.50d.zip"
)

DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "glove.2024.wikigiga.50d.zip"

# Download the GloVe archive once and return its local path
@lru_cache(maxsize=1)
def download_zip() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print("Downloading GloVe from Stanford...")
        urllib.request.urlretrieve(
            GLOVE_URL,
            ZIP_PATH,
        )
        print("Download complete!")

    else:
        print("Using cached GloVe zip file...")

    return str(ZIP_PATH)


def extract_portion(
    lines_to_keep: int | None,
    output_name: str,
    zip_path: str,
) -> str:
    """
    Extract either the first N lines or the complete GloVe file.

    Parameters
    ----------
    lines_to_keep:
        Number of vocabulary lines to extract.
        None means extract the entire file.

    output_name:
        Name of the output file without extension.

    zip_path:
        Path to the downloaded GloVe zip archive.
    """

    output_file = DATA_DIR / f"{output_name}.txt"

    if lines_to_keep is None:
        print(f"Extracting full GloVe file to {output_file}...")
    else:
        print(
            f"Extracting {lines_to_keep:,} lines "
            f"to {output_file}..."
        )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        txt_files = [
            name
            for name in zip_ref.namelist()
            if name.endswith(".txt")
        ]

        if not txt_files:
            raise FileNotFoundError(
                "No .txt file found inside the GloVe archive."
            )

        txt_file_name = txt_files[0]

        with zip_ref.open(txt_file_name) as infile:
            with open(output_file, "wb") as outfile:

                for i, line in enumerate(infile):

                    if (
                        lines_to_keep is not None
                        and i >= lines_to_keep
                    ):
                        break

                    outfile.write(line)

    print(f"Saved file: {output_file}")

    return str(output_file)


def extract_all_sizes():
    """
    Extract the full GloVe file and all five supported
    web vocabulary sizes.
    """

    zip_path = download_zip()

    sizes = {
        "glove.2024.wikigiga.50d_500": 500,
        "glove.2024.wikigiga.50d_1000": 1000,
        "glove.2024.wikigiga.50d_2500": 2500,
        "glove.2024.wikigiga.50d_5000": 5000,
        "glove.2024.wikigiga.50d_full": None,
    }

    for name, size in sizes.items():

        output_file = DATA_DIR / f"{name}.txt"

        if output_file.exists():
            print(f"Already exists: {output_file}")
            continue

        extract_portion(
            lines_to_keep=size,
            output_name=name,
            zip_path=zip_path,
        )


if __name__ == "__main__":
    extract_all_sizes()