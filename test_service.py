#!/usr/bin/env python3
"""
Test script for the GPU Service (CatVTON + DreamO).

Run on the GPU machine to verify everything works before exposing via ngrok.

Usage:
    # 1. Check GPU & dependencies (no server needed)
    python test_service.py --check-gpu

    # 2. Test the running service (start main.py first)
    python test_service.py --test-api

    # 3. Test DreamO-specific endpoints (twin generation)
    python test_service.py --test-dreamo

    # 4. Test with a real try-on job (uses sample images from the web)
    python test_service.py --test-tryon

    # 5. Run all checks
    python test_service.py --all

    # Test against a remote URL
    python test_service.py --all --url https://xxxx.ngrok-free.app
"""
from __future__ import annotations

import argparse
import json
import sys
import time

# ─── GPU & dependency checks (no server needed) ─────────────────────────────

def check_gpu():
    """Verify CUDA, NVIDIA GPU, and required packages are available."""
    print("=" * 60)
    print("GPU & DEPENDENCY CHECK")
    print("=" * 60)

    errors = []

    # 1. PyTorch + CUDA
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"[OK] CUDA available — {count} device(s)")
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                vram = props.total_memory // 1024**2
                print(f"     GPU {i}: {props.name} ({vram} MB VRAM)")
                if vram < 4000:
                    errors.append(f"GPU {i} has only {vram} MB VRAM (need >= 4 GB)")
                if vram < 16000:
                    print(f"     [WARN] GPU {i}: DreamO int8 needs >=16 GB VRAM. CatVTON OK.")

            # Quick tensor test on GPU
            t = torch.randn(2, 2, device="cuda")
            _ = t @ t
            print("[OK] CUDA tensor compute works")
        else:
            errors.append("CUDA is NOT available — model will run on CPU (very slow)")
            print("[!!] CUDA not available")
    except ImportError:
        errors.append("PyTorch not installed")
        print("[FAIL] PyTorch not installed")

    # 2. diffusers
    try:
        import diffusers
        print(f"[OK] diffusers {diffusers.__version__}")
    except ImportError:
        errors.append("diffusers not installed")
        print("[FAIL] diffusers not installed")

    # 3. transformers
    try:
        import transformers
        print(f"[OK] transformers {transformers.__version__}")
    except ImportError:
        errors.append("transformers not installed")
        print("[FAIL] transformers not installed")

    # 4. accelerate
    try:
        import accelerate
        print(f"[OK] accelerate {accelerate.__version__}")
    except ImportError:
        errors.append("accelerate not installed")
        print("[FAIL] accelerate not installed")

    # 5. FastAPI + uvicorn
    try:
        import fastapi
        print(f"[OK] fastapi {fastapi.__version__}")
    except ImportError:
        errors.append("fastapi not installed")
        print("[FAIL] fastapi not installed")

    try:
        import uvicorn
        print("[OK] uvicorn installed")
    except ImportError:
        errors.append("uvicorn not installed")
        print("[FAIL] uvicorn not installed")

    # 6. PIL + requests
    try:
        from PIL import Image  # noqa: F401
        print("[OK] Pillow installed")
    except ImportError:
        errors.append("Pillow not installed")
        print("[FAIL] Pillow not installed")

    try:
        import requests
        print(f"[OK] requests {requests.__version__}")
    except ImportError:
        errors.append("requests not installed")
        print("[FAIL] requests not installed")

    # 7. DreamO (optional but recommended)
    try:
        from dreamo.dreamo_pipeline import DreamOPipeline  # noqa: F401
        print("[OK] DreamO package installed")
    except ImportError:
        print("[WARN] DreamO not installed (needed for INFERENCE_BACKEND=dreamo)")
        print("       Install: pip install git+https://github.com/bytedance/DreamO.git")

    # Summary
    print()
    if errors:
        print("ISSUES FOUND:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("All checks passed! Ready to run the service.")
        return True


# ─── API tests (server must be running) ─────────────────────────────────────

def test_api(base_url: str):
    """Test the running service endpoints."""
    import requests

    print("=" * 60)
    print(f"API TEST — {base_url}")
    print("=" * 60)

    errors = []

    # 1. Health endpoint
    print("\n1. GET /health")
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   [OK] Status: {data['status']}")
            print(f"   Backend: {data.get('inference_backend', 'unknown')}")
            print(f"   Model: {data['model_id']}")
            print(f"   Device: {data['device']}")
            print(f"   Dtype: {data.get('dtype', 'unknown')}")
            print(f"   Pipeline loaded: {data['pipeline_loaded']}")
            gpu = data.get("gpu", {})
            if gpu.get("cuda_available"):
                print(f"   GPU: {gpu['device_name']} ({gpu['vram_total_mb']} MB)")
                print(f"   VRAM allocated: {gpu['vram_allocated_mb']} MB / reserved: {gpu['vram_reserved_mb']} MB")
            else:
                print("   [!!] CUDA not available on service")
                errors.append("CUDA not available")
        else:
            errors.append(f"/health returned {resp.status_code}")
            print(f"   [FAIL] HTTP {resp.status_code}")
    except requests.ConnectionError:
        errors.append(f"Cannot connect to {base_url} — is the service running?")
        print(f"   [FAIL] Connection refused — is `python main.py` running?")
        print()
        return False

    # 2. Status 404 (expected for unknown job)
    print("\n2. GET /status/nonexistent")
    try:
        resp = requests.get(f"{base_url}/status/nonexistent", timeout=10)
        if resp.status_code == 404:
            print("   [OK] Returns 404 for unknown job (expected)")
        else:
            errors.append(f"/status/nonexistent returned {resp.status_code} instead of 404")
            print(f"   [FAIL] Expected 404, got {resp.status_code}")
    except Exception as e:
        errors.append(str(e))
        print(f"   [FAIL] {e}")

    # 3. Try-on with invalid URL (should accept job, then fail gracefully)
    print("\n3. POST /try-on (with invalid image URLs)")
    try:
        resp = requests.post(
            f"{base_url}/try-on",
            json={
                "model_image_url": "http://invalid.test/person.jpg",
                "garment_image_url": "http://invalid.test/garment.jpg",
            },
            timeout=10,
        )
        if resp.status_code == 202:
            job_id = resp.json()["job_id"]
            print(f"   [OK] Job accepted: {job_id}")

            # Wait for it to fail
            time.sleep(5)
            status_resp = requests.get(f"{base_url}/status/{job_id}", timeout=10)
            status_data = status_resp.json()
            if status_data["status"] == "error":
                print(f"   [OK] Job failed gracefully: {status_data['error'][:80]}")
            elif status_data["status"] == "processing":
                print("   [OK] Job still processing (will eventually fail)")
            else:
                print(f"   [??] Unexpected status: {status_data['status']}")
        else:
            errors.append(f"/try-on returned {resp.status_code}")
            print(f"   [FAIL] HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        errors.append(str(e))
        print(f"   [FAIL] {e}")

    # 4. Validation — missing fields
    print("\n4. POST /try-on (missing fields)")
    try:
        resp = requests.post(f"{base_url}/try-on", json={}, timeout=10)
        if resp.status_code == 422:
            print("   [OK] Returns 422 for missing fields (expected)")
        else:
            errors.append(f"Missing-field request returned {resp.status_code} instead of 422")
            print(f"   [FAIL] Expected 422, got {resp.status_code}")
    except Exception as e:
        errors.append(str(e))
        print(f"   [FAIL] {e}")

    # Summary
    print()
    if errors:
        print("ISSUES FOUND:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("All API tests passed!")
        return True


# ─── DreamO-specific tests (twin generation, DreamO health) ─────────────────

def test_dreamo(base_url: str):
    """Test DreamO-specific functionality (twin generation endpoint)."""
    import requests

    print("=" * 60)
    print(f"DREAMO-SPECIFIC TEST — {base_url}")
    print("=" * 60)

    errors = []

    # 1. Check backend is DreamO
    print("\n1. Verify inference backend")
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        if resp.status_code != 200:
            print(f"   [FAIL] /health returned {resp.status_code}")
            return False
        data = resp.json()
        backend = data.get("inference_backend", "unknown")
        print(f"   Backend: {backend}")
        if backend != "dreamo":
            print(f"   [SKIP] Service is running '{backend}', not 'dreamo'.")
            print("          Twin generation requires INFERENCE_BACKEND=dreamo.")
            print("          Skipping DreamO-specific tests.")
            return True  # not a failure, just not applicable
        print("   [OK] DreamO backend confirmed")
    except requests.ConnectionError:
        print(f"   [FAIL] Cannot connect to {base_url}")
        return False

    # 2. Twin generation — missing fields → 422
    print("\n2. POST /generate-twin (missing fields)")
    try:
        resp = requests.post(f"{base_url}/generate-twin", json={}, timeout=10)
        if resp.status_code == 422:
            print("   [OK] Returns 422 for missing fields (expected)")
        else:
            errors.append(f"Missing-field twin request returned {resp.status_code} instead of 422")
            print(f"   [FAIL] Expected 422, got {resp.status_code}")
    except Exception as e:
        errors.append(str(e))
        print(f"   [FAIL] {e}")

    # 3. Twin generation — invalid URL (graceful failure)
    print("\n3. POST /generate-twin (with invalid face URL)")
    try:
        resp = requests.post(
            f"{base_url}/generate-twin",
            json={"face_image_url": "http://invalid.test/face.jpg"},
            timeout=10,
        )
        if resp.status_code == 202:
            job_id = resp.json()["job_id"]
            print(f"   [OK] Twin job accepted: {job_id}")

            time.sleep(5)
            status_resp = requests.get(f"{base_url}/status/{job_id}", timeout=10)
            status_data = status_resp.json()
            if status_data["status"] == "error":
                print(f"   [OK] Job failed gracefully: {status_data['error'][:80]}")
            elif status_data["status"] == "processing":
                print("   [OK] Job still processing (will eventually fail)")
            else:
                print(f"   [??] Unexpected status: {status_data['status']}")
        else:
            errors.append(f"/generate-twin returned {resp.status_code}")
            print(f"   [FAIL] HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        errors.append(str(e))
        print(f"   [FAIL] {e}")

    # 4. Try-on with DreamO-specific params
    print("\n4. POST /try-on (with DreamO params: category, prompt)")
    try:
        resp = requests.post(
            f"{base_url}/try-on",
            json={
                "model_image_url": "http://invalid.test/person.jpg",
                "garment_image_url": "http://invalid.test/garment.jpg",
                "category": "tops",
                "prompt": "A person wearing this top, photorealistic",
                "num_steps": 8,
                "guidance_scale": 3.5,
            },
            timeout=10,
        )
        if resp.status_code == 202:
            print(f"   [OK] Job accepted with DreamO params: {resp.json()['job_id']}")
        else:
            errors.append(f"/try-on with DreamO params returned {resp.status_code}")
            print(f"   [FAIL] HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        errors.append(str(e))
        print(f"   [FAIL] {e}")

    # Summary
    print()
    if errors:
        print("ISSUES FOUND:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("All DreamO tests passed!")
        return True


# ─── Full try-on test (uses real sample images) ─────────────────────────────

# Public-domain sample images (replace with real person/garment for meaningful output)
SAMPLE_PERSON_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
SAMPLE_GARMENT_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
SAMPLE_FACE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"


def _poll_job(base_url: str, job_id: str, label: str, timeout_s: int = 900) -> bool:
    """Poll a job until done/error/timeout. Returns True on success."""
    import requests

    start = time.time()
    while time.time() - start < timeout_s:
        status_resp = requests.get(f"{base_url}/status/{job_id}", timeout=10)
        data = status_resp.json()
        elapsed = int(time.time() - start)

        if data["status"] == "done":
            print(f"\n[OK] {label} completed in {elapsed}s")
            print(f"     Result: {data['image_url']}")
            return True
        elif data["status"] == "error":
            print(f"\n[FAIL] {label} failed after {elapsed}s: {data['error']}")
            return False
        else:
            print(f"  ... {label} processing ({elapsed}s elapsed)", end="\r", flush=True)
            time.sleep(10)

    print(f"\n[FAIL] {label} timed out after {timeout_s}s")
    return False


def test_tryon(base_url: str):
    """Submit a real try-on job and wait for the result."""
    import requests

    print("=" * 60)
    print(f"TRY-ON INFERENCE TEST — {base_url}")
    print("=" * 60)
    print()
    print("NOTE: This test verifies the full pipeline runs without crashing.")
    print("The output image won't be meaningful (sample images aren't real")
    print("person/garment photos). Replace the URLs below for a real test.")
    print()
    print(f"Person URL:  {SAMPLE_PERSON_URL}")
    print(f"Garment URL: {SAMPLE_GARMENT_URL}")
    print()

    # Submit try-on job
    try:
        resp = requests.post(
            f"{base_url}/try-on",
            json={
                "model_image_url": SAMPLE_PERSON_URL,
                "garment_image_url": SAMPLE_GARMENT_URL,
            },
            timeout=10,
        )
    except requests.ConnectionError:
        print(f"[FAIL] Cannot connect to {base_url}")
        return False

    if resp.status_code != 202:
        print(f"[FAIL] HTTP {resp.status_code}: {resp.text}")
        return False

    job_id = resp.json()["job_id"]
    print(f"Try-on job submitted: {job_id}")
    print("Waiting for inference (this may take 10s–10min depending on GPU)...")

    return _poll_job(base_url, job_id, "Try-on")


def test_twin(base_url: str):
    """Submit a twin generation job and wait for the result (DreamO only)."""
    import requests

    print("=" * 60)
    print(f"TWIN GENERATION TEST — {base_url}")
    print("=" * 60)
    print()

    # Check backend
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        data = resp.json()
        if data.get("inference_backend") != "dreamo":
            print("[SKIP] Twin generation requires INFERENCE_BACKEND=dreamo")
            return True
    except requests.ConnectionError:
        print(f"[FAIL] Cannot connect to {base_url}")
        return False

    print(f"Face URL: {SAMPLE_FACE_URL}")
    print()

    # Submit twin job
    try:
        resp = requests.post(
            f"{base_url}/generate-twin",
            json={"face_image_url": SAMPLE_FACE_URL},
            timeout=10,
        )
    except requests.ConnectionError:
        print(f"[FAIL] Cannot connect to {base_url}")
        return False

    if resp.status_code != 202:
        print(f"[FAIL] HTTP {resp.status_code}: {resp.text}")
        return False

    job_id = resp.json()["job_id"]
    print(f"Twin job submitted: {job_id}")
    print("Waiting for inference...")

    return _poll_job(base_url, job_id, "Twin generation")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test the GPU Service (CatVTON + DreamO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_service.py --check-gpu
  python test_service.py --test-api
  python test_service.py --test-dreamo
  python test_service.py --test-tryon
  python test_service.py --all
  python test_service.py --all --url https://xxxx.ngrok-free.app
        """,
    )
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU & dependencies")
    parser.add_argument("--test-api", action="store_true", help="Test API endpoints (service must be running)")
    parser.add_argument("--test-dreamo", action="store_true", help="Test DreamO-specific endpoints (twin, params)")
    parser.add_argument("--test-tryon", action="store_true", help="Run a real try-on inference job")
    parser.add_argument("--test-twin", action="store_true", help="Run a real twin generation job (DreamO only)")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--url", default="http://localhost:8001", help="Service base URL (default: http://localhost:8001)")
    args = parser.parse_args()

    if not any([args.check_gpu, args.test_api, args.test_dreamo, args.test_tryon, args.test_twin, args.all]):
        parser.print_help()
        sys.exit(0)

    results = []

    if args.check_gpu or args.all:
        results.append(("GPU Check", check_gpu()))
        print()

    if args.test_api or args.all:
        results.append(("API Test", test_api(args.url)))
        print()

    if args.test_dreamo or args.all:
        results.append(("DreamO Test", test_dreamo(args.url)))
        print()

    if args.test_tryon or args.all:
        results.append(("Try-On Test", test_tryon(args.url)))
        print()

    if args.test_twin or args.all:
        results.append(("Twin Gen Test", test_twin(args.url)))
        print()

    # Final summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
