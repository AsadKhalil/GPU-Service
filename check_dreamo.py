#!/usr/bin/env python3
"""
Verify DreamO import the same way main.py does.

``python -c "import dreamo.dreamo_pipeline"`` only adds the *current working
directory* to sys.path, so it works from inside the DreamO clone but fails from
GPU-Service. This script runs dreamo_inference's path logic first.

Usage (from gpu_service / GPU-Service):

  python check_dreamo.py

Or with explicit clone root:

  DREAMO_SRC=/path/to/DreamO python check_dreamo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# gpu_service directory (where this file lives)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dreamo_inference as di  # noqa: E402

di._ensure_dreamo_path()
import dreamo.dreamo_pipeline  # noqa: E402, F401

print("dreamo OK")
