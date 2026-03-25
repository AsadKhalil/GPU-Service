from __future__ import annotations

import logging
import sys
import uuid
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image

import config as cfg

logger = logging.getLogger(__name__)

_pipeline = None


def _dreamo_import_ok() -> bool:
    try:
        import importlib

        importlib.import_module("dreamo.dreamo_pipeline")
        return True
    except ImportError:
        return False


def _ensure_dreamo_path() -> None:
    """
    ByteDance/DreamO's GitHub repo has no pip install metadata (pyproject.toml is
    only Ruff/Black). ``pip install git+...`` does not register ``dreamo``.
    We must put the *clone root* (the folder that contains ``dreamo/``) on
    ``sys.path`` — not the inner ``dreamo`` folder.
    """
    if _dreamo_import_ok():
        return

    here = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if cfg.DREAMO_SRC:
        candidates.append(Path(cfg.DREAMO_SRC).expanduser().resolve())
    candidates.extend(
        [
            here / "DreamO",
            here / "dreamo-main",
            here.parent / "DreamO",
            here.parent / "dreamo-main",
            here.parent / "dreamo",
        ]
    )

    seen: set[Path] = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        marker = root / "dreamo" / "dreamo_pipeline.py"
        if not marker.is_file():
            continue
        s = str(root)
        inserted = False
        if s not in sys.path:
            sys.path.insert(0, s)
            inserted = True
        if _dreamo_import_ok():
            logger.info("DreamO: using repo at %s", root)
            return
        if inserted:
            sys.path.remove(s)

    hint = (
        "DreamO is not pip-installable from GitHub (no [project] in pyproject.toml). "
        "Clone the repo and point DREAMO_SRC at the directory that contains "
        "the inner package folder `dreamo/` (i.e. .../DreamO/dreamo/dreamo_pipeline.py).\n"
        "  export DREAMO_SRC=/absolute/path/to/DreamO\n"
        "Do NOT set DREAMO_SRC to .../DreamO/dreamo (one level too deep).\n"
        f"Tried clone roots: {candidates}"
    )
    raise ImportError(hint)


def _load_pipeline():
    """Load the DreamO pipeline (ByteDance/DreamO on FLUX.1-dev)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading DreamO pipeline from %s …", cfg.DREAMO_MODEL_ID)

    _ensure_dreamo_path()

    from dreamo.dreamo_pipeline import DreamOPipeline

    pipe = DreamOPipeline.from_pretrained(
        cfg.DREAMO_BASE_MODEL,
        torch_dtype=cfg.DTYPE,
    )
    pipe.load_dreamo_lora(cfg.DREAMO_MODEL_ID)

    quant = cfg.DREAMO_QUANTIZE
    if quant == "int8":
        pipe.enable_model_cpu_offload()
    elif quant == "nunchaku":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(cfg.DEVICE)

    _pipeline = pipe
    logger.info("DreamO pipeline loaded (quantize=%s).", quant)
    return _pipeline


def _fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def run_tryon(
    person_image_url: str,
    garment_image_url: str,
    *,
    category: str = "auto",
    prompt: str = "",
    num_steps: int | None = None,
    guidance_scale: float | None = None,
) -> Path:
    """
    Virtual try-on using DreamO.

    Uses the person image as an ID reference (face/body preservation)
    and the garment image as an IP reference (garment transfer).
    """
    pipe = _load_pipeline()

    person_img = _fetch_image(person_image_url)
    garment_img = _fetch_image(garment_image_url)

    steps = num_steps or cfg.DREAMO_NUM_STEPS
    gs = guidance_scale or cfg.DREAMO_GUIDANCE_SCALE

    if not prompt:
        category_prompts = {
            "tops": "A person wearing this top, full body, high quality, photorealistic",
            "bottoms": "A person wearing these pants, full body, high quality, photorealistic",
            "one-pieces": "A person wearing this outfit, full body, high quality, photorealistic",
            "auto": "A person wearing this garment, full body, high quality, photorealistic",
        }
        prompt = category_prompts.get(category, category_prompts["auto"])

    with torch.inference_mode():
        result = pipe.generate_image(
            prompt=prompt,
            ref_images=[person_img, garment_img],
            ref_tasks=["id", "ip"],
            num_inference_steps=steps,
            guidance_scale=gs,
            width=cfg.DREAMO_WIDTH,
            height=cfg.DREAMO_HEIGHT,
        )

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.RESULTS_DIR / f"{uuid.uuid4()}.png"
    result.save(out_path)
    logger.info("DreamO result saved to %s (%dx%d, %d steps)", out_path, cfg.DREAMO_WIDTH, cfg.DREAMO_HEIGHT, steps)
    return out_path


def run_twin_generation(
    face_image_url: str,
    *,
    prompt: str = "",
    num_steps: int | None = None,
    guidance_scale: float | None = None,
) -> Path:
    """
    Generate a digital twin / avatar image using DreamO's ID preservation mode.
    """
    pipe = _load_pipeline()

    face_img = _fetch_image(face_image_url)

    steps = num_steps or cfg.DREAMO_NUM_STEPS
    gs = guidance_scale or cfg.DREAMO_GUIDANCE_SCALE

    if not prompt:
        prompt = (
            "A full-body portrait of this person standing straight, "
            "neutral pose, arms slightly away from body, "
            "plain white background, high quality, photorealistic"
        )

    with torch.inference_mode():
        result = pipe.generate_image(
            prompt=prompt,
            ref_images=[face_img],
            ref_tasks=["id"],
            num_inference_steps=steps,
            guidance_scale=gs,
            width=cfg.DREAMO_WIDTH,
            height=cfg.DREAMO_HEIGHT,
        )

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.RESULTS_DIR / f"{uuid.uuid4()}.png"
    result.save(out_path)
    logger.info("DreamO twin result saved to %s", out_path)
    return out_path
