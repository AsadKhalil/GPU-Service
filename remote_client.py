#!/usr/bin/env python3
"""
Send local images to a remote GPU service (FastAPI main.py), wait for the job,
and save the result image locally.

The GPU worker downloads inputs with requests.get(), so your laptop paths are
not visible to it. This script can:

  * Upload files via --upload-backend (default: catbox). 0x0.st often returns
    403 for some ISPs/datacenters; use catbox, litterbox, transfer, or imgbb.
  * Or pass --person-url / --garment-url / --face-url if images are already
    on a URL the GPU can reach (S3, Django media, ngrok to your PC).

Set BASE_URL on the GPU machine to a URL your client can use to download
/results/... (ngrok or public IP:port). If the API returns localhost links,
this script rewrites the download URL to --gpu-url.

Examples:

  # DreamO digital twin (one face photo)
  python remote_client.py twin \\
    --gpu-url http://127.0.0.1:8001 \\
    --face api_tests/input/person.jpg \\
    -o ./out/avatar.png

  # DreamO try-on (person + garment)
  python remote_client.py try-on \\
    --gpu-url https://xxxx.ngrok-free.app \\
    --person ./person.jpg --garment ./top.jpeg \\
    -o ./out/tryon.png

  # Already-public URLs (no upload)
  python remote_client.py try-on \\
    --gpu-url http://GPU:8001 \\
    --person-url https://example.com/a.jpg \\
    --garment-url https://example.com/b.jpg \\
    -o out.png
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

DEFAULT_UPLOAD_BACKEND = "catbox"
URL_0X0_DEFAULT = "https://0x0.st"
POLL_INTERVAL_SEC = 2.0
POLL_MAX_SEC = 1800  # 30 min for heavy DreamO loads

UPLOAD_BACKENDS = (
    "catbox",
    "litterbox",
    "0x0",
    "transfer",
    "imgbb",
)


def _ensure_http_url(text: str, context: str) -> str:
    url = (text or "").strip()
    if not url.startswith("http"):
        raise RuntimeError(f"Unexpected {context} response: {text[:200]!r}")
    return url


def upload_0x0(path: Path, upload_url: str = URL_0X0_DEFAULT) -> str:
    """POST file to 0x0.st (often bans VPN/hosting ranges)."""
    with path.open("rb") as f:
        r = requests.post(
            upload_url,
            files={"file": (path.name, f)},
            timeout=300,
        )
    r.raise_for_status()
    return _ensure_http_url(r.text, "0x0")


def upload_catbox(path: Path) -> str:
    """Anonymous permanent file on catbox.moe."""
    with path.open("rb") as f:
        r = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": (path.name, f)},
            timeout=300,
        )
    r.raise_for_status()
    return _ensure_http_url(r.text, "catbox")


def upload_litterbox(path: Path, keep: str = "24h") -> str:
    """Temporary file on litterbox.catbox.moe (keep: 1h, 12h, 24h, 72h)."""
    with path.open("rb") as f:
        r = requests.post(
            "https://litterbox.catbox.moe/resources/internals/api.php",
            data={"reqtype": "fileupload", "time": keep},
            files={"fileToUpload": (path.name, f)},
            timeout=300,
        )
    r.raise_for_status()
    return _ensure_http_url(r.text, "litterbox")


def upload_transfer_sh(path: Path) -> str:
    """PUT to transfer.sh (availability varies by region)."""
    data = path.read_bytes()
    r = requests.put(
        f"https://transfer.sh/{path.name}",
        data=data,
        timeout=300,
    )
    r.raise_for_status()
    return _ensure_http_url(r.text, "transfer.sh")


def upload_imgbb(path: Path, api_key: str) -> str:
    """ImgBB — set IMGBB_API_KEY in the environment."""
    b64 = base64.b64encode(path.read_bytes()).decode()
    r = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": api_key, "image": b64},
        timeout=300,
    )
    r.raise_for_status()
    payload = r.json()
    if not payload.get("success") or "data" not in payload:
        raise RuntimeError(f"imgbb error: {payload}")
    return _ensure_http_url(payload["data"]["url"], "imgbb")


def upload_local_file(path: Path, backend: str, upload_url: str) -> str:
    if backend == "catbox":
        return upload_catbox(path)
    if backend == "litterbox":
        return upload_litterbox(path)
    if backend == "0x0":
        return upload_0x0(path, upload_url)
    if backend == "transfer":
        return upload_transfer_sh(path)
    if backend == "imgbb":
        key = os.environ.get("IMGBB_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "upload-backend imgbb requires IMGBB_API_KEY in the environment",
            )
        return upload_imgbb(path, key)
    raise ValueError(f"unknown upload backend: {backend}")


def resolve_download_url(image_url: str, gpu_base: str) -> str:
    """If API returned localhost BASE_URL, fetch via --gpu-url instead."""
    gpu_base = gpu_base.rstrip("/")
    iu = urlparse(image_url)
    host = (iu.hostname or "").lower()
    if host in ("localhost", "127.0.0.1"):
        pu = urlparse(gpu_base)
        return urlunparse(
            (pu.scheme, pu.netloc, iu.path, "", iu.query, iu.fragment)
        )
    return image_url


def poll_until_done(base: str, job_id: str) -> dict:
    base = base.rstrip("/")
    deadline = time.monotonic() + POLL_MAX_SEC
    while time.monotonic() < deadline:
        r = requests.get(f"{base}/status/{job_id}", timeout=60)
        r.raise_for_status()
        data = r.json()
        st = data.get("status")
        if st == "done":
            return data
        if st == "error":
            raise RuntimeError(data.get("error") or "job failed")
        time.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(f"Job {job_id} did not finish within {POLL_MAX_SEC}s")


def save_image(url: str, out: Path) -> None:
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)


def cmd_twin(ns: argparse.Namespace) -> int:
    base = ns.gpu_url.rstrip("/")
    if ns.face_url:
        face_url = ns.face_url
    else:
        face_path = Path(ns.face).expanduser().resolve()
        if not face_path.is_file():
            print(f"Not a file: {face_path}", file=sys.stderr)
            return 1
        print(f"Uploading face image ({ns.upload_backend}) …")
        face_url = upload_local_file(
            face_path, ns.upload_backend, ns.upload_url,
        )

    payload = {"face_image_url": face_url}
    if ns.prompt:
        payload["prompt"] = ns.prompt

    print(f"POST {base}/generate-twin …")
    r = requests.post(f"{base}/generate-twin", json=payload, timeout=30)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"job_id={job_id}, waiting …")
    result = poll_until_done(base, job_id)
    img_url = result.get("image_url")
    if not img_url:
        raise RuntimeError("done but no image_url")
    dl = resolve_download_url(img_url, base)
    print(f"Downloading → {ns.output}")
    save_image(dl, Path(ns.output).expanduser().resolve())
    print("OK")
    return 0


def cmd_tryon(ns: argparse.Namespace) -> int:
    base = ns.gpu_url.rstrip("/")
    if ns.person_url:
        p_url = ns.person_url
    else:
        p = Path(ns.person).expanduser().resolve()
        if not p.is_file():
            print(f"Not a file: {p}", file=sys.stderr)
            return 1
        print(f"Uploading person image ({ns.upload_backend}) …")
        p_url = upload_local_file(p, ns.upload_backend, ns.upload_url)

    if ns.garment_url:
        g_url = ns.garment_url
    else:
        g = Path(ns.garment).expanduser().resolve()
        if not g.is_file():
            print(f"Not a file: {g}", file=sys.stderr)
            return 1
        print(f"Uploading garment image ({ns.upload_backend}) …")
        g_url = upload_local_file(g, ns.upload_backend, ns.upload_url)

    payload = {
        "model_image_url": p_url,
        "garment_image_url": g_url,
        "category": ns.category,
    }
    if ns.prompt:
        payload["prompt"] = ns.prompt

    print(f"POST {base}/try-on …")
    r = requests.post(f"{base}/try-on", json=payload, timeout=30)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"job_id={job_id}, waiting …")
    result = poll_until_done(base, job_id)
    img_url = result.get("image_url")
    if not img_url:
        raise RuntimeError("done but no image_url")
    dl = resolve_download_url(img_url, base)
    print(f"Downloading → {ns.output}")
    save_image(dl, Path(ns.output).expanduser().resolve())
    print("OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Local client for GPU try-on / twin service",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser(
        "twin",
        help="DreamO digital twin / avatar (POST /generate-twin)",
    )
    pt.add_argument(
        "--gpu-url",
        required=True,
        help="GPU service base, e.g. http://IP:8001",
    )
    pt.add_argument("--face", help="Local face/person image path")
    pt.add_argument("--face-url", help="Public URL (skips upload)")
    pt.add_argument("--prompt", default="", help="Optional prompt override")
    pt.add_argument(
        "--upload-backend",
        choices=UPLOAD_BACKENDS,
        default=DEFAULT_UPLOAD_BACKEND,
        help="Host for temporary public image URLs (GPU fetches these)",
    )
    pt.add_argument(
        "--upload-url",
        default=URL_0X0_DEFAULT,
        help="Only used with --upload-backend 0x0",
    )
    pt.add_argument(
        "-o",
        "--output",
        required=True,
        help="Local path to save PNG/JPEG",
    )
    pt.set_defaults(func=cmd_twin)

    po = sub.add_parser("try-on", help="Virtual try-on (POST /try-on)")
    po.add_argument("--gpu-url", required=True)
    po.add_argument("--person", help="Local person / model photo")
    po.add_argument("--person-url", help="Public URL for person image")
    po.add_argument("--garment", help="Local garment photo")
    po.add_argument("--garment-url", help="Public URL for garment")
    po.add_argument("--category", default="auto")
    po.add_argument("--prompt", default="")
    po.add_argument(
        "--upload-backend",
        choices=UPLOAD_BACKENDS,
        default=DEFAULT_UPLOAD_BACKEND,
    )
    po.add_argument("--upload-url", default=URL_0X0_DEFAULT)
    po.add_argument("-o", "--output", required=True)
    po.set_defaults(func=cmd_tryon)

    ns = p.parse_args()
    if ns.cmd == "twin" and not (ns.face or ns.face_url):
        p.error("twin: provide --face or --face-url")
    if ns.cmd == "try-on":
        if not (ns.person or ns.person_url):
            p.error("try-on: provide --person or --person-url")
        if not (ns.garment or ns.garment_url):
            p.error("try-on: provide --garment or --garment-url")

    try:
        return ns.func(ns)
    except requests.RequestException as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        if getattr(e, "response", None) is not None:
            print(e.response.text[:500], file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
