from __future__ import annotations

import os
from pathlib import Path

import torch

# ── GPU selection ────────────────────────────────────────────────────────────
# On a dual-GPU machine (AMD + NVIDIA), CUDA only sees NVIDIA GPUs.
# Set CUDA_VISIBLE_DEVICES to pin to a specific NVIDIA card (e.g. "0").
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_ID = os.getenv("MODEL_ID", "zhengchong/CatVTON")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
RESULTS_DIR = Path("results")
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8001"))

# ── Active inference backend ─────────────────────────────────────────────────
# "catvton" (default) or "dreamo"
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "catvton").lower()

# ── DreamO settings ──────────────────────────────────────────────────────────
DREAMO_MODEL_ID = os.getenv("DREAMO_MODEL_ID", "ByteDance/DreamO")
DREAMO_BASE_MODEL = os.getenv("DREAMO_BASE_MODEL", "black-forest-labs/FLUX.1-dev")
DREAMO_QUANTIZE = os.getenv("DREAMO_QUANTIZE", "int8")  # none | int8 | nunchaku
DREAMO_NUM_STEPS = int(os.getenv("DREAMO_NUM_STEPS", "12"))
DREAMO_GUIDANCE_SCALE = float(os.getenv("DREAMO_GUIDANCE_SCALE", "3.5"))
DREAMO_WIDTH = int(os.getenv("DREAMO_WIDTH", "768"))
DREAMO_HEIGHT = int(os.getenv("DREAMO_HEIGHT", "1024"))

# Cloned DreamO without `pip install -e .`? Set to repo root (dir containing
# the `dreamo/` package folder).
DREAMO_SRC = os.getenv("DREAMO_SRC", "").strip()

# ── Base URL for result links ────────────────────────────────────────────────
# When using ngrok, set BASE_URL to your ngrok URL (e.g. https://abc123.ngrok-free.app)
# so that result image URLs are accessible externally.
BASE_URL = os.getenv("BASE_URL", f"http://localhost:{PORT}")
