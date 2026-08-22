"""Shared pytest fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the src/ layout package is importable without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# All tests use the mock GPU backend — no NVML required.
os.environ.setdefault("GPUPROF_MOCK", "1")


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")
