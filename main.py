from __future__ import annotations

import logging
import threading
import uuid
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config as cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Stylish GPU Service")

# In-memory job store (single-user local use)
jobs: dict[str, dict] = {}


# ── Static file serving ──────────────────────────────────────────────────────

cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/results",
    StaticFiles(directory=str(cfg.RESULTS_DIR)),
    name="results",
)


# ── Inference backend ────────────────────────────────────────────────────────

def _get_inference():
    """Return the active inference module based on INFERENCE_BACKEND."""
    if cfg.INFERENCE_BACKEND == "dreamo":
        import dreamo_inference as mod
    else:
        import inference as mod
    return mod


# ── Schemas ──────────────────────────────────────────────────────────────────

class TryOnRequest(BaseModel):
    model_image_url: str
    garment_image_url: str
    category: str = "auto"
    prompt: str = ""
    num_steps: int | None = None
    guidance_scale: float | None = None


class TwinRequest(BaseModel):
    face_image_url: str
    prompt: str = ""
    num_steps: int | None = None
    guidance_scale: float | None = None


class TryOnResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    status: Literal["processing", "done", "error"]
    image_url: str | None = None
    video_url: str | None = None
    error: str | None = None


# ── Background workers ───────────────────────────────────────────────────────

def _run_tryon_job(job_id: str, body: TryOnRequest) -> None:
    jobs[job_id]["status"] = "processing"
    try:
        mod = _get_inference()
        if cfg.INFERENCE_BACKEND == "dreamo":
            result_path = mod.run_tryon(
                body.model_image_url,
                body.garment_image_url,
                category=body.category,
                prompt=body.prompt,
                num_steps=body.num_steps,
                guidance_scale=body.guidance_scale,
            )
        else:
            result_path = mod.run_tryon(
                body.model_image_url,
                body.garment_image_url,
            )
        image_url = f"{cfg.BASE_URL}/results/{result_path.name}"
        jobs[job_id].update({"status": "done", "image_url": image_url})
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        jobs[job_id].update({"status": "error", "error": str(exc)})


def _run_twin_job(job_id: str, body: TwinRequest) -> None:
    jobs[job_id]["status"] = "processing"
    try:
        mod = _get_inference()
        if cfg.INFERENCE_BACKEND != "dreamo":
            raise ValueError("Twin generation requires INFERENCE_BACKEND=dreamo")
        result_path = mod.run_twin_generation(
            body.face_image_url,
            prompt=body.prompt,
            num_steps=body.num_steps,
            guidance_scale=body.guidance_scale,
        )
        image_url = f"{cfg.BASE_URL}/results/{result_path.name}"
        jobs[job_id].update({"status": "done", "image_url": image_url})
    except Exception as exc:
        logger.exception("Twin job %s failed: %s", job_id, exc)
        jobs[job_id].update({"status": "error", "error": str(exc)})


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check service status and GPU info."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2),
            "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2),
            "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2),
        }
    else:
        gpu_info = {"cuda_available": False, "device_count": 0}

    mod = _get_inference()
    return {
        "status": "ok",
        "inference_backend": cfg.INFERENCE_BACKEND,
        "model_id": cfg.DREAMO_MODEL_ID if cfg.INFERENCE_BACKEND == "dreamo" else cfg.MODEL_ID,
        "device": cfg.DEVICE,
        "dtype": str(cfg.DTYPE),
        "base_url": cfg.BASE_URL,
        "pipeline_loaded": mod._pipeline is not None,
        "gpu": gpu_info,
    }


@app.post("/try-on", response_model=TryOnResponse, status_code=202)
def start_tryon(body: TryOnRequest) -> TryOnResponse:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "image_url": None,
        "video_url": None,
    }
    thread = threading.Thread(
        target=_run_tryon_job,
        args=(job_id, body),
        daemon=True,
    )
    thread.start()
    return TryOnResponse(job_id=job_id)


@app.post("/generate-twin", response_model=TryOnResponse, status_code=202)
def start_twin(body: TwinRequest) -> TryOnResponse:
    """Generate a digital twin avatar using DreamO's ID preservation mode."""
    if cfg.INFERENCE_BACKEND != "dreamo":
        raise HTTPException(
            status_code=400,
            detail="Twin generation requires INFERENCE_BACKEND=dreamo",
        )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "image_url": None,
        "video_url": None,
    }
    thread = threading.Thread(
        target=_run_twin_job,
        args=(job_id, body),
        daemon=True,
    )
    thread.start()
    return TryOnResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str) -> StatusResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="not found")
    return StatusResponse(
        status=job["status"],
        image_url=job.get("image_url"),
        video_url=None,
        error=job.get("error"),
    )


if __name__ == "__main__":
    logger.info(
        "Starting GPU Service on %s:%s (backend=%s)",
        cfg.HOST, cfg.PORT, cfg.INFERENCE_BACKEND,
    )
    logger.info("BASE_URL = %s (set BASE_URL env var for ngrok)", cfg.BASE_URL)
    logger.info("CUDA available: %s | Device: %s", torch.cuda.is_available(), cfg.DEVICE)
    if torch.cuda.is_available():
        logger.info(
            "GPU: %s | VRAM: %d MB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory // 1024**2,
        )
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT)
