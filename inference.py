from __future__ import annotations

import logging
import uuid
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image

import config as cfg

logger = logging.getLogger(__name__)

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading CatVTON pipeline from %s …", cfg.MODEL_ID)
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(
        cfg.MODEL_ID,
        torch_dtype=cfg.DTYPE,
    )
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    _pipeline = pipe
    logger.info("Pipeline loaded.")
    return _pipeline


def _fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def run_tryon(person_image_url: str, garment_image_url: str) -> Path:
    """
    Download images, run CatVTON inference, save result PNG, return its path.
    """
    pipe = _load_pipeline()

    person_img = _fetch_image(person_image_url)
    garment_img = _fetch_image(garment_image_url)

    # CatVTON uses the garment as the inpainting reference image.
    # A white mask covering the full body area is used so the model
    # composites the garment onto the person.
    mask = Image.new("RGB", person_img.size, (255, 255, 255))

    with torch.inference_mode():
        output = pipe(
            prompt="a person wearing the garment, high quality, photorealistic",
            image=person_img,
            mask_image=mask,
            ip_adapter_image=garment_img,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.RESULTS_DIR / f"{uuid.uuid4()}.png"
    output.save(out_path)
    logger.info("Saved result to %s", out_path)
    return out_path
