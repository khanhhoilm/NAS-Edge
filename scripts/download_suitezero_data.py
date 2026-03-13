"""Download NAS-Bench-Suite-Zero (NASLib) dataset files.

SuiteZero's `nasbench/nas_bench_suite_zero/naslib/utils/get_dataset_api.py` expects the
following files under:
  nasbench/nas_bench_suite_zero/naslib/data/

- nasbench_only108.pkl
- nb201_cifar10_full_training.pickle
- nb201_cifar100_full_training.pickle
- nb201_ImageNet16_full_training.pickle

Those files are hosted in a Google Drive folder referenced by SuiteZero.
This script downloads that folder using `gdown`.

Usage:
  python scripts/download_suitezero_data.py

Notes:
- Requires internet access.
- Requires `gdown` (installed automatically if missing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


GDRIVE_FOLDER_ID = "1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa"

REQUIRED_FILES = [
    "nasbench_only108.pkl",
    "nb201_cifar10_full_training.pickle",
    "nb201_cifar100_full_training.pickle",
    "nb201_ImageNet16_full_training.pickle",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _repo_root() / "nasbench" / "nas_bench_suite_zero" / "naslib" / "data"


def _ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401

        return
    except Exception:
        pass

    print("`gdown` not found; installing into current Python env...", file=sys.stderr)
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])  # nosec B603,B607


def _missing_files(data_dir: Path) -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_FILES:
        if not (data_dir / name).exists():
            missing.append(name)
    return missing


def main() -> int:
    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    missing_before = _missing_files(data_dir)
    if not missing_before:
        print(f"All required SuiteZero files already present in: {data_dir}")
        return 0

    _ensure_gdown()

    import gdown

    print(f"Downloading Google Drive folder {GDRIVE_FOLDER_ID} -> {data_dir}")

    # gdown supports downloading shared folders by ID.
    # This will download all contents of the folder.
    gdown.download_folder(
        id=GDRIVE_FOLDER_ID,
        output=str(data_dir),
        quiet=False,
        use_cookies=False,
    )

    missing_after = _missing_files(data_dir)
    if missing_after:
        print("Downloaded folder, but some required files are still missing:")
        for name in missing_after:
            print(f"- {name}")
        print(
            "If the Drive folder contents changed, download the missing files manually "
            f"into: {data_dir}",
            file=sys.stderr,
        )
        return 2

    print("SuiteZero dataset files downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
