"""gpuprof server: FastAPI ingest + live dashboard.

Run with::

    pip install -e ".[server]"
    python -m gpuprof.server --host 0.0.0.0 --port 8000 --api-key SECRET
"""
from .app import create_app

__all__ = ["create_app"]
