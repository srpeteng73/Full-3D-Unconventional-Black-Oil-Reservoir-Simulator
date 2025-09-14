# tests/test_smoke.py
# Goal: make sure key files are syntactically valid WITHOUT importing Streamlit.

import ast
import os

FILES_TO_CHECK = [
    "app.py",
    "full3d.py",
    "engines/implicit.py",
    "engines/fast.py",
]

def parse_ok(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        ast.parse(f.read(), filename=path)

def test_python_files_parse():
    for p in FILES_TO_CHECK:
        assert os.path.exists(p), f"Missing file: {p}"
        parse_ok(p)
