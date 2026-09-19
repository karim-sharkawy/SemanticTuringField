"""Generate all Semantic Turing Field GIF visualizations.

This keeps the export pipeline centralized in the scripts directory while
writing the resulting files into a dedicated assets/gifs folder.

Examples:
    python scripts/generate_all_gifs.py
    python scripts/generate_all_gifs.py --only demo
    python scripts/generate_all_gifs.py --only dt --output-dir assets/gifs
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "gifs"

GENERATORS = {
    "demo": ("src.generate_demo_gif", "stf_demo.gif"),
    "dt": ("src.generate_dt_gif", "dt_comparison.gif"),
    "sentence": ("src.generate_sentence_gif", "sentence_response.gif"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Semantic Turing Field GIFs in a dedicated assets/gifs folder.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(GENERATORS),
        help="Generate only one GIF family instead of all of them.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where output GIFs should be stored.",
    )
    return parser.parse_args()


def generate_gif(module_name: str, output_filename: str, output_dir: Path) -> None:
    module = importlib.import_module(module_name)
    module.OUTPUT_PATH = output_dir / output_filename
    print(f"\n=== Generating {module_name} ===")
    module.main()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [args.only] if args.only else list(GENERATORS)

    for name in selected:
        module_name, output_filename = GENERATORS[name]
        generate_gif(module_name, output_filename, output_dir)


if __name__ == "__main__":
    main()
