"""
main.py — FastAPI Colorization Backend (Model-as-a-Service)
===========================================================
Production-grade REST API serving the ECCV 2016 deep colorization model
on a CUDA GPU. Implements the complete L→ab inference pipeline with
enterprise-grade lifecycle management, error handling, and telemetry.

Endpoints:
    GET  /health     — System health check (GPU status, model state, VRAM)
    POST /translate   — Colorize a grayscale image via multipart upload

Server Lifecycle:
    - Model loads ONCE during startup via FastAPI lifespan handler
    - Weights auto-download from S3 on first run (~130MB)
    - Model pinned to GPU in eval mode with gradients globally disabled
    - CUDA cache cleared on shutdown
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from colorizer import eccv16, preprocess_img, postprocess_tens
from metrics import InferenceProfiler, get_system_snapshot

# ─── Logging Configuration ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-24s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("colorization.server")


# ─── Custom Exception ─────────────────────────────────────────────────────

class ColorizationError(Exception):
    """Domain-specific exception for colorization pipeline failures."""

    def __init__(self, message: str, stage: str, detail: str = "") -> None:
        self.message = message
        self.stage = stage
        self.detail = detail
        super().__init__(self.message)


# ─── Global Model State ───────────────────────────────────────────────────
# Populated during lifespan startup; never reassigned during serving.

_model: torch.nn.Module | None = None
_device: torch.device = torch.device("cpu")


# ─── Lifespan Handler ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan handler for model lifecycle management.

    Startup:
        1. Detect compute device (CUDA preferred, CPU fallback)
        2. Instantiate ECCVGenerator with pretrained S3 weights
        3. Pin model to GPU memory in eval mode
        4. Globally disable gradient computation

    Shutdown:
        1. Delete model reference
        2. Flush CUDA memory caches
    """
    global _model, _device

    # ── Device Selection ──
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        logger.info("CUDA device detected: %s", torch.cuda.get_device_name(0))
        logger.info(
            "VRAM available: %.1f MB",
            torch.cuda.get_device_properties(0).total_mem / (1024 ** 2),
        )
    else:
        _device = torch.device("cpu")
        logger.warning("No CUDA device found — falling back to CPU inference")

    # ── Model Loading ──
    logger.info("Loading ECCVGenerator with pretrained weights from S3...")
    _model = eccv16(pretrained=True)
    _model.to(_device)
    _model.eval()
    torch.set_grad_enabled(False)

    param_count = sum(p.numel() for p in _model.parameters())
    logger.info(
        "Model loaded — %s parameters | Device: %s",
        f"{param_count:,}",
        _device,
    )

    yield  # ── Server is running ──

    # ── Cleanup ──
    logger.info("Shutting down — releasing model and CUDA memory")
    del _model
    _model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── FastAPI Application ──────────────────────────────────────────────────

app = FastAPI(
    title="ChromaSynth: Image Colorization Engine",
    description=(
        "Production MaaS endpoint for automatic grayscale→RGB colorization "
        "using the ECCV 2016 deep network (Zhang et al.)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Global Exception Handler ─────────────────────────────────────────────

@app.exception_handler(ColorizationError)
async def colorization_error_handler(request, exc: ColorizationError):
    """Return structured error payload for domain exceptions."""
    logger.error("Pipeline failure at [%s]: %s — %s", exc.stage, exc.message, exc.detail)
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.message,
            "stage": exc.stage,
            "detail": exc.detail,
        },
    )


# ─── Health Endpoint ──────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """System health check with GPU diagnostics.

    Returns:
        JSON object with model status, device info, and resource utilization.
    """
    snapshot = get_system_snapshot()
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "device": str(_device),
        "gpu": snapshot.gpu_name,
        "gpu_allocated_mb": snapshot.gpu_allocated_mb,
        "gpu_reserved_mb": snapshot.gpu_reserved_mb,
        "ram_used_mb": round(snapshot.ram_used_mb, 1),
        "cpu_percent": snapshot.cpu_percent,
    }


# ─── Colorization Endpoint ────────────────────────────────────────────────

@app.post("/translate", tags=["Inference"])
async def translate(file: UploadFile = File(...)) -> StreamingResponse:
    """Colorize a grayscale image using the ECCV 2016 deep network.

    Pipeline Stages:
        1. DECODE     — Read multipart bytes → OpenCV BGR matrix
        2. VALIDATE   — Shape/channel checks, RGBA→RGB, grayscale→3ch
        3. PREPROCESS — RGB→Lab, extract L channel, resize to 256×256 tensor
        4. INFERENCE  — GPU forward pass through ECCVGenerator (L→ab)
        5. POSTPROCESS— Concatenate L+ab, Lab→RGB, upsample to original size
        6. ENCODE     — RGB→JPEG byte stream at quality 92

    Args:
        file: Multipart-encoded image file (.png, .jpg, .jpeg).

    Returns:
        StreamingResponse with JPEG content-type and profiling headers.

    Raises:
        HTTPException: 400 for invalid images, 503 if model not loaded.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # ── Stage 1: DECODE ──
    try:
        raw_bytes = await file.read()
        np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {str(e)}",
        )

    # ── Stage 2: VALIDATE & NORMALIZE ──
    try:
        # Handle RGBA → RGB (drop alpha channel)
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
            logger.info("Converted RGBA→BGR (alpha channel dropped)")

        # Handle true grayscale → 3-channel
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            logger.info("Converted single-channel grayscale→BGR")

        # BGR → RGB for skimage Lab conversion
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Validate minimum dimensions
        h, w = img_rgb.shape[:2]
        if h < 8 or w < 8:
            raise ValueError(f"Image too small: {w}×{h} (minimum 8×8)")

        logger.info("Input validated — shape: %d×%d×%d", h, w, img_rgb.shape[2])

    except ValueError:
        raise
    except Exception as e:
        raise ColorizationError(
            message="Image validation failed",
            stage="VALIDATE",
            detail=str(e),
        )

    # ── Stage 3: PREPROCESS (RGB → Lab → L tensor) ──
    try:
        tens_orig_l, tens_rs_l = preprocess_img(img_rgb, HW=(256, 256))
        logger.info(
            "Preprocessed — orig_l: %s | resized_l: %s",
            list(tens_orig_l.shape),
            list(tens_rs_l.shape),
        )
    except Exception as e:
        raise ColorizationError(
            message="Lab preprocessing failed",
            stage="PREPROCESS",
            detail=str(e),
        )

    # ── Stage 4: GPU INFERENCE ──
    profiler = InferenceProfiler()
    try:
        with profiler:
            tens_rs_l_gpu = tens_rs_l.to(_device)
            out_ab = _model(tens_rs_l_gpu)

        logger.info(
            "Inference complete — ab shape: %s | latency: %.1fms",
            list(out_ab.shape),
            profiler.result.latency_ms,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise ColorizationError(
                message="GPU out of memory",
                stage="INFERENCE",
                detail="Try a smaller image or restart the server",
            )
        raise ColorizationError(
            message="GPU inference failed",
            stage="INFERENCE",
            detail=str(e),
        )

    # ── Stage 5: POSTPROCESS (L + ab → Lab → RGB) ──
    try:
        out_ab_cpu = out_ab.cpu()
        rgb_out = postprocess_tens(tens_orig_l, out_ab_cpu)

        # Clip to valid range and convert to uint8
        rgb_out_uint8 = (np.clip(rgb_out, 0, 1) * 255).astype(np.uint8)

        logger.info("Postprocessed — output shape: %s", list(rgb_out_uint8.shape))
    except Exception as e:
        raise ColorizationError(
            message="Lab→RGB postprocessing failed",
            stage="POSTPROCESS",
            detail=str(e),
        )

    # ── Stage 6: ENCODE (RGB → JPEG byte stream) ──
    try:
        # Convert RGB → BGR for OpenCV encoding
        bgr_out = cv2.cvtColor(rgb_out_uint8, cv2.COLOR_RGB2BGR)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
        success, jpeg_buf = cv2.imencode(".jpg", bgr_out, encode_params)
        if not success:
            raise RuntimeError("cv2.imencode returned failure status")

        jpeg_bytes = jpeg_buf.tobytes()
        logger.info("Encoded JPEG — %d bytes (quality=92)", len(jpeg_bytes))
    except Exception as e:
        raise ColorizationError(
            message="JPEG encoding failed",
            stage="ENCODE",
            detail=str(e),
        )

    # ── Response with profiling headers ──
    headers = profiler.result.to_headers()
    headers["X-Original-Size"] = f"{w}x{h}"
    headers["X-Model-Device"] = str(_device)

    return StreamingResponse(
        io.BytesIO(jpeg_bytes),
        media_type="image/jpeg",
        headers=headers,
    )


# ─── Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=4500,
        log_level="info",
        workers=1,  # Single worker — model is loaded per-process
    )
