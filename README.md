# GPU Service — Virtual Try-On & Digital Twin

FastAPI service that runs inference on a GPU machine. Supports two backends:

| Backend | Model | Use case | Min VRAM |
|---------|-------|----------|----------|
| **DreamO** (recommended) | [ByteDance/DreamO](https://huggingface.co/ByteDance/DreamO) on FLUX.1-dev | Try-on + digital twin generation | 16 GB (int8), 24 GB (fp16) |
| CatVTON | [zhengchong/CatVTON](https://huggingface.co/zhengchong/CatVTON) | Try-on only | 4 GB |

---

## Cloud GPU Setup (DreamO)

Tested on: RunPod, Vast.ai, Lambda Cloud, Google Colab Pro+ (A100/L40S/RTX 4090).

### 1. Provision a GPU instance

Pick an instance with **16 GB+ VRAM** and a recent NVIDIA driver (535+):

| GPU | VRAM | int8 inference | fp16 inference |
|-----|------|----------------|----------------|
| RTX 4090 | 24 GB | ~15s/image | ~10s/image |
| A100 40G | 40 GB | ~8s/image | ~6s/image |
| L40S | 48 GB | ~7s/image | ~5s/image |
| RTX 3090 | 24 GB | ~20s/image | ~15s/image |
| T4 (16 GB) | 16 GB | ~45s/image (int8 only) | OOM |

Use a PyTorch base image if available (e.g. `runpod/pytorch:2.1.0-py3.10-cuda12.1.0`).

### 2. Install dependencies

```bash
# SSH into the GPU instance
ssh user@<gpu-ip>

# Clone the repo (or just copy gpu_service/)
git clone <repo-url> && cd <repo>/gpu_service

# Create a venv (recommended)
python3 -m venv .venv && source .venv/bin/activate

# Install base dependencies
pip install -r requirements.txt

# Install DreamO
pip install git+https://github.com/bytedance/DreamO.git
```

> **Note:** The DreamO package pulls in its own version of diffusers. If you hit
> version conflicts, install DreamO first, then the rest of requirements.txt.

### 3. Verify GPU

```bash
python test_service.py --check-gpu
```

Should report CUDA available, 16 GB+ VRAM, and all packages installed.

### 4. Start the service

```bash
# Basic start (DreamO backend, int8 quantization)
INFERENCE_BACKEND=dreamo python main.py

# With custom settings
INFERENCE_BACKEND=dreamo \
DREAMO_QUANTIZE=none \
DREAMO_NUM_STEPS=20 \
DREAMO_WIDTH=768 \
DREAMO_HEIGHT=1024 \
python main.py
```

First run downloads ~12 GB of model weights (FLUX.1-dev base + DreamO LoRA).

The service listens on `0.0.0.0:8001`.

### 5. Expose to the internet

**Option A — ngrok (quick)**

```bash
# In a second terminal/tmux pane
ngrok http 8001

# Note the URL: https://xxxx.ngrok-free.app
# Restart with BASE_URL so result image links work:
BASE_URL=https://xxxx.ngrok-free.app INFERENCE_BACKEND=dreamo python main.py
```

**Option B — RunPod/Vast.ai port forwarding**

Most cloud GPU providers expose ports automatically. Use the assigned public URL
as `BASE_URL`.

**Option C — Cloudflare Tunnel**

```bash
cloudflared tunnel --url http://localhost:8001
```

### 6. Configure Django backend

In `backend/.env`:

```dotenv
AI_TRYON_PROVIDER=DREAMO
DREAMO_GPU_HOST=https://xxxx.ngrok-free.app   # your public GPU service URL
```

Restart Django + Celery. The backend now routes try-on requests to your cloud GPU.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_BACKEND` | `catvton` | `dreamo` or `catvton` |
| `PORT` | `8001` | Service port |
| `BASE_URL` | `http://localhost:8001` | Public URL for result image links |
| `CUDA_VISIBLE_DEVICES` | `0` | Which NVIDIA GPU to use |
| `DREAMO_MODEL_ID` | `ByteDance/DreamO` | HuggingFace model ID for DreamO LoRA |
| `DREAMO_BASE_MODEL` | `black-forest-labs/FLUX.1-dev` | Base model for DreamO |
| `DREAMO_QUANTIZE` | `int8` | `none` (fp16), `int8` (CPU offload), `nunchaku` (sequential offload) |
| `DREAMO_NUM_STEPS` | `12` | Inference steps (more = better quality, slower) |
| `DREAMO_GUIDANCE_SCALE` | `3.5` | CFG scale |
| `DREAMO_WIDTH` | `768` | Output image width |
| `DREAMO_HEIGHT` | `1024` | Output image height |

---

## API Endpoints

### `GET /health`

Returns service status, GPU info, loaded model, VRAM usage.

### `POST /try-on` → `202 Accepted`

```json
{
  "model_image_url": "https://...",
  "garment_image_url": "https://...",
  "category": "tops",
  "prompt": "",
  "num_steps": 12,
  "guidance_scale": 3.5
}
```

Returns `{"job_id": "uuid"}`. Poll `/status/{job_id}` for the result.

### `POST /generate-twin` → `202 Accepted` (DreamO only)

```json
{
  "face_image_url": "https://...",
  "prompt": "",
  "num_steps": 12,
  "guidance_scale": 3.5
}
```

Generates a full-body digital twin from a face photo using ID preservation.

### `GET /status/{job_id}`

```json
{
  "status": "processing|done|error",
  "image_url": "https://.../results/uuid.png",
  "video_url": null,
  "error": null
}
```

---

## Dual-GPU Machine (AMD + NVIDIA)

- CUDA only sees NVIDIA GPUs — AMD cards are ignored automatically.
- `CUDA_VISIBLE_DEVICES` defaults to `"0"` (first NVIDIA GPU).
- Check the startup log to confirm the correct GPU is selected.
- For multiple NVIDIA GPUs: `CUDA_VISIBLE_DEVICES=1` selects the second one.

---

## Testing

```bash
# Check GPU & dependencies (no server needed)
python test_service.py --check-gpu

# Test API endpoints (start main.py first)
python test_service.py --test-api

# Test DreamO-specific endpoints (twin generation)
python test_service.py --test-dreamo

# Run a full try-on inference test (takes 10s–5min depending on GPU)
python test_service.py --test-tryon

# Run everything
python test_service.py --all

# Test against a remote URL (ngrok, RunPod, etc.)
python test_service.py --all --url https://xxxx.ngrok-free.app
```

### Quick curl test

```bash
# Health check
curl http://localhost:8001/health | python -m json.tool

# Submit a try-on job
curl -s -X POST http://localhost:8001/try-on \
  -H 'Content-Type: application/json' \
  -d '{"model_image_url":"https://example.com/person.jpg",
       "garment_image_url":"https://example.com/shirt.jpg"}' | python -m json.tool

# Submit a twin generation job (DreamO only)
curl -s -X POST http://localhost:8001/generate-twin \
  -H 'Content-Type: application/json' \
  -d '{"face_image_url":"https://example.com/face.jpg"}' | python -m json.tool

# Check job status
curl http://localhost:8001/status/<job_id> | python -m json.tool
```

---

## Resource Usage

| Backend | VRAM | RAM | Time per image | Disk |
|---------|------|-----|----------------|------|
| DreamO (int8) | 10–16 GB | 16–24 GB | 10–45s (GPU dependent) | ~12 GB weights |
| DreamO (fp16) | 16–24 GB | 12–16 GB | 5–20s (GPU dependent) | ~12 GB weights |
| CatVTON | 4–5 GB | 8–12 GB | 3–8 min (RTX 2060) | ~7 GB weights |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `OutOfMemoryError` | Use `DREAMO_QUANTIZE=int8` or reduce `DREAMO_WIDTH`/`DREAMO_HEIGHT` |
| `CUDA not available` | Check `nvidia-smi` works; reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| Model download hangs | Set `HF_HUB_ENABLE_HF_TRANSFER=1` and `pip install hf_transfer` for faster downloads |
| ngrok free tier limits | Use `BASE_URL` env var; consider Cloudflare Tunnel as a free alternative |
| `Twin generation requires INFERENCE_BACKEND=dreamo` | Set `INFERENCE_BACKEND=dreamo` — twin gen only works with DreamO |
| Slow first request | Expected — model loads into VRAM on first inference. Subsequent requests are faster. |
