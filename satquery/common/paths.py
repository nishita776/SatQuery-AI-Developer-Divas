"""Repo-level paths. Override with env vars when running outside the repo (e.g. on Kaggle)."""

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("SATQUERY_ROOT", Path(__file__).resolve().parents[2]))


def fixtures_dir() -> Path:
    return Path(os.environ.get("SATQUERY_FIXTURES_DIR", REPO_ROOT / "fixtures"))


def fake_results_dir() -> Path:
    return fixtures_dir() / "fake_results"


def cache_dir() -> Path:
    return Path(os.environ.get("SATQUERY_CACHE_DIR", REPO_ROOT / "cache"))


def registry_dir() -> Path:
    return Path(os.environ.get("SATQUERY_REGISTRY_DIR", REPO_ROOT / "registry"))
