#!/usr/bin/env python3
"""
Download pretrained diabetes models from HuggingFace to the models/ directory.
Source repo: https://huggingface.co/Neperl/diabetes-health-indicators-study
"""

import os
import sys
import shutil
import logging
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_model")

REPO_ID = os.getenv("MODEL_REPO_ID", "Neperl/diabetes-health-indicators-study")

# Map HuggingFace artifact names directly to canonical application names
FILENAME_MAPPING = {
    "diabetes_random_forest_tuned.joblib": "v1.joblib",
    "diabetes_mlp_tuned.keras": "v2.keras",
    "scaler.joblib": "scaler.joblib",
}


def get_models_dir() -> Path:
    env_dir = os.getenv("MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    # Check container path
    if Path("/app/models").exists() or Path("/app").is_dir():
        return Path("/app/models")
    # Check host local path
    return Path(__file__).resolve().parent.parent.parent / "models"


def download_models():
    """Download model artifacts from HuggingFace directly to models/ without redundant files."""
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory: {models_dir}")
    logger.info(f"Connecting to HuggingFace repository: {REPO_ID}")

    try:
        repo_files = list_repo_files(REPO_ID)
        logger.info(f"Found files in repo: {repo_files}")
    except Exception as exc:
        logger.error(f"Failed to query repository: {exc}")
        sys.exit(1)

    downloaded = []
    # Use a temporary directory for downloading to prevent cache residue in models/
    with tempfile.TemporaryDirectory() as tmp_dir:
        for filename, target_name in FILENAME_MAPPING.items():
            if filename in repo_files:
                logger.info(f"Downloading {filename} as {target_name}...")
                downloaded_file = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    local_dir=tmp_dir,
                    local_dir_use_symlinks=False,
                )
                destination = models_dir / target_name
                shutil.move(downloaded_file, destination)
                downloaded.append(str(destination))
                logger.info(f"✅ Saved: {destination}")

    logger.info(f"Successfully downloaded {len(downloaded)} model artifacts to {models_dir}")


if __name__ == "__main__":
    download_models()
