"""Backward-compatible entry point for the v3 personal survival pipeline."""

from __future__ import annotations

from .build_dataset import main as build_dataset
from .train_baseline import main as train_baseline


def main() -> None:
    build_dataset()
    train_baseline()


if __name__ == "__main__":
    main()
