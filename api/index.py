import base64
import io
import os
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageEnhance
from openai import OpenAI
import openai

# Enable HEIC/HEIF decoding (if installed successfully)
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
    HEIF_ENABLED = True
except Exception:
    HEIF_ENABLED = False

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Models ---
# Use the dedicated image edit endpoint with GPT Image models
# Docs: /v1/images/edits supports gpt-image-1, gpt-image-1-mini, gpt-image-1.5 (if your org has access) :contentReference[oaicite:2]{index=2}
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")  # safest default
# If your org actually has it, you can set IMAGE_MODEL=gpt-image-1.5

# --- Vercel limits ---
# Vercel Functions request+response payload max is 4.5MB. Keep margin. :contentReference[oaicite:3]{index=3}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))  # 4MB

# --- Image caps ---
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(4000 * 4000)))  # 16MP

# --- Deterministic tuning (safe, low hallucination) ---
WB_STRENGTH = float(os.getenv("WB_STRENGTH", "0.90"))          # 0..1
CONTRAST_GAIN = float(os.getenv("CONTRAST_GAIN", "1.12"))
COLOR_GAIN = float(os.getenv("COLOR_GAIN", "1.07"))
BRIGHTNESS_GAIN = float(os.getenv("BRIGHTNESS_GAIN", "1.02"))
SHARPNESS_GAIN = float(os.getenv("SHARPNESS_GAIN", "1.14"))

# --- Anti-warp (optional but recommended) ---
EDGE_PROTECT_ON = os.getenv("EDGE_PROTECT_ON", "true").lower() == "true"
EDGE_PROTECT_ALPHA = float(os.getenv("EDGE_PROTECT_ALPHA", "0.80"))  # 0..1
EDGE_THRESH = float(os.getenv("EDGE_THRESH", "0.20"))               # 0..1

PROMPT = """
Edit the provided photo (do NOT generate a new one).
Goal: professional car listing enhancement.

STRICT: keep framing identical (no crop/zoom/rotate).
DO NOT change: wheels/rims/tires/spokes, center-cap logos, badges/emblems, grille design, headlights/taillights,
any text/icons/buttons/screens (must remain crisp/readable), trims (no new chrome), geometry, or add/remove objects.

ONLY global improvements:
- neutralize color cast / white balance
- natural exposure and contrast
- mild highlight recovery, mild shadow lift
- subtle clarity/sharpness (no halos)
- subtle noise reduction (do not smear texture)
Photorealistic.
""".strip()


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc(), "path": str(request.url)},
    )


def load_rgb(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
    except Exception as e:
        if not HEIF_ENABLED:
            raise HTTPException(
                status_code=400,
                detail="Cannot open this image. If it's HEIC/HEIF, pillow-heif may not be active on this deploy.",
            ) from e
        raise HTTPException(status_code=400, detail="Cannot open image.") from e

    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def downscale_if_needed(im: Image.Image) -> Image.Image:
    w, h = im.size
    if (w * h) <= MAX_PIXELS:
        return im
    scale = (MAX_PIXELS / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def gray_world_wb(im: Image.Image, strength: float) -> Image.Image:
    arr = np.asarray(im).astype(np.float32) / 255.0
    means = arr.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = float(means.mean())
    gains = gray / means
    gains = 1.0 + strength * (gains - 1.0)
    out = np.clip(arr * gains[None, None, :], 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8), mode="RGB")


def tone_curve(im: Image.Image) -> Image.Image:
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = arr
    out = np.clip((x - 0.5) * 1.18 + 0.5, 0, 1)
    out = out ** 0.985
    return Image.fromarray((out * 255).astype(np.uint8), mode="RGB")


def deterministic_pass(im: Image.Image) -> Image.Image:
    im = gray_world_wb(im, strength=WB_STRENGTH)
    im = tone_curve(im)
    im = ImageEnhance.Contrast(im).enhance(CONTRAST_GAIN)
    im = ImageEnhance.Color(im).enhance(COLOR_GAIN)
    im = ImageEnhance.Brightness(im).enhance(BRIGHTNESS_GAIN)
    im = ImageEnhance.Sharpness(im).enhance(SHARPNESS_GAIN)
    return im


def to_png_bytes(im: Image.Image, compress_level: int = 6) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


def edge_mask(im: Image.Image, thresh: float) -> np.ndarray:
    g = np.asarray(im.convert("L")).astype(np.float32) / 255.0
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5

    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-6)
    m = (mag > thresh).astype(np.float32)

    # tiny blur (3x3 box)
    k = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]], dtype=np.float32) / 9.0
    p = np.pad(m, ((1, 1), (1, 1)), mode="edge")
    m2 = np.empty_like(m)
    for y in range(m.shape[0]):
        for x in range(m.shape[1]):
            m2[y, x] = float((p[y:y + 3, x:x + 3] * k).sum())

    return np.clip(m2, 0, 1)


def protect_edges(base: Image.Image, ai: Image.Image, alpha: float, thresh: float) -> Image.Image:
    if ai.size != base.size:
        ai = ai.resize(base.size, Image.Resampling.LANCZOS)

    m = edge_mask(base, thresh=thresh)
    m = np.clip(m * alpha, 0, 1)
    m3 = m[..., None]

    base_np = np.asarray(base).astype(np.float32)
    ai_np = np.asarray(ai).astype(np.float32)

    out = ai_np * (1.0 - m3) + base_np * m3
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


@app.get("/")
def root():
    return {"ok": True}


@app.get("/version")
def version():
    return {
        "openai_version": getattr(openai, "__version__", "unknown"),
        "image_model": IMAGE_MODEL,
        "heif_enabled": HEIF_ENABLED,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_pixels": MAX_PIXELS,
        "edge_protect_on": EDGE_PROTECT_ON,
        "edge_protect_alpha": EDGE_PROTECT_ALPHA,
        "edge_thresh": EDGE_THRESH,
    }


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")

    # Avoid Vercel 413 (Function payload limit) :contentReference[oaicite:4]{index=4}
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large for Vercel Functions (~4.5MB limit). Got {len(data)} bytes. "
                   f"Use smaller test uploads or move to URL-based uploads in production.",
        )

    # 1) Decode + normalize
    original = downscale_if_needed(load_rgb(data))

    # 2) Deterministic “pro” pass (guaranteed improvements)
    base = deterministic_pass(original)

    # Convert to PNG for GPT image models (png/webp/jpg) :contentReference[oaicite:5]{index=5}
    base_png = to_png_bytes(base, compress_level=6)

    # 3) AI edit (DEDICATED edits endpoint)
    # size="auto" to preserve orientation/sizing rules per model. :contentReference[oaicite:6]{index=6}
    kwargs = dict(
        model=IMAGE_MODEL,
        image=[("input.png", base_png)],
        prompt=PROMPT,
        size="auto",
        quality="high",
        output_format="png",
    )

    # input_fidelity is only supported for gpt-image-1 (not mini) :contentReference[oaicite:7]{index=7}
    if IMAGE_MODEL == "gpt-image-1":
        kwargs["input_fidelity"] = "high"

    result = client.images.edit(**kwargs)

    b64 = result.data[0].b64_json
    out_bytes = base64.b64decode(b64)

    # 4) Optional anti-warp: restore sharp edges from deterministic base
    if EDGE_PROTECT_ON:
        ai_im = Image.open(io.BytesIO(out_bytes)).convert("RGB")
        fixed = protect_edges(base, ai_im, alpha=EDGE_PROTECT_ALPHA, thresh=EDGE_THRESH)
        out_bytes = to_png_bytes(fixed, compress_level=6)

    return Response(
        content=out_bytes,
        media_type="image/png",
        headers={
            "x-ai-used": "true",
            "x-image-model": IMAGE_MODEL,
            "x-edge-protect": str(EDGE_PROTECT_ON).lower(),
            "x-edge-alpha": str(EDGE_PROTECT_ALPHA),
            "x-edge-thresh": str(EDGE_THRESH),
            "cache-control": "no-store",
        },
    )
