#!/usr/bin/env python3
from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path


_BASE_DIR = Path(__file__).resolve().parent
_CANDIDATES = [
    _BASE_DIR / "__pycache__" / "isic2019_aligned_loader.cpython-313.pyc",
]
_PYC_PATH = next((path for path in _CANDIDATES if path.exists()), None)
if _PYC_PATH is None:
    raise FileNotFoundError("Missing original isic2019_aligned_loader pyc payload")

_LOADER = importlib.machinery.SourcelessFileLoader(__name__ + ".__shim__", str(_PYC_PATH))
_SPEC = importlib.util.spec_from_loader(__name__ + ".__shim__", _LOADER)
if _SPEC is None:
    raise RuntimeError(f"Failed to load spec for {_PYC_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(_MODULE)
globals().update(_MODULE.__dict__)
