# tests/test_smoke.py

import importlib

def test_core_imports():
    """Very light smoke test: key modules import without error."""
    for m in ("core.full3d", "engines.fast", "engines.implicit"):
        importlib.import_module(m)
