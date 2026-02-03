import base64
import io
import os
import traceback
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageEnhance
from openai import OpenAI
import openai

# Optional HEIC/HEIF support
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
    HEIF_ENABLED = True
except Exception:
    HEIF_ENABLED = False


app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAIN_MODEL = os.getenv("MAIN_MODEL", "gpt-4.1")

# Vercel Function hard request/response payload limit is 4.5MB. Keep headroom. :contentReference[oaicite:4]{index=4}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))  # 4MB

# Image processing caps
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(4000 * 4000)))  # 16MP
# Edge-protect tunables
EDGE_PROTECT_ALPHA = float(os.getenv("EDGE_PROTECT_ALPHA", "0.80"))  # 0..1
EDGE_THRESH = float(os.getenv("EDGE_THRESH", "0.20"))               # 0..1

# AI strength: if you want more polish, slightly increase these deterministic gains,
# not the AI. That keeps hallucinations down.
WB_STRENGTH = float(os.getenv("WB_STRENGTH", "0.90"))   # 0..1
CONTRAST_GAIN = float(os.getenv("CONTRAST_GAIN", "1.10"))
COLOR_GAIN = float(os.getenv("COLOR_GAIN", "1.06"))
BRIGHTNESS_GAIN = float(os.getenv("BRIGHTNESS_GAIN", "1.02"))
SHARPNESS_GAIN = float(os.getenv("SHARPNESS_GAIN", "1.12"))

PROMPT = """
Edit the provided photo (do NOT generate a new one).
Goal: professional car listing enhancement.

Keep framing identical. No crop/zoom/rotate.
Do not change wheels/rims/center-cap logos, badges, grille design, headlights/taillights.
Do not alter or warp any text/icons/buttons/screens; keep them crisp and readable.
No new chrome/trim, no geometry changes, no object add/remove.

Only global improvements: neutralize color cast, natural contrast/exposure, mild highlight recovery,
mild shadow lift, subtle clarity/sharpness, subtle noise reduction. Photorealistic.
""".strip()


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc(), "path": str(request.url)},
    )


def _load_rgb(data: bytes) -> Image.Image:
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


def _downscale_if_needed(im: Image.Image) -> Image.Image:
    w, h = im.size
    if (w * h) <= MAX_PIXELS:
        return im
    scale = (MAX_PIXELS / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _pick_tool_size(w: int, h: int) -> str:
    # Use highest supported sizes to preserve details (logos/text).
    if w > h:
        return "1536x1024"
    if h > w:
        return "1024x1536"
    return "1024x1024"


def _to_data_url_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _decode_b64_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ---------------------------
# Deterministic pro correction
# ---------------------------

def _gray_world_wb(im: Image.Image, strength: float) -> Image.Image:
    arr = np.asarray(im).astype(np.float32) / 255.0
    means = arr.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = float(means.mean())
    gains = gray / means
    gains = 1.0 + strength * (gains - 1.0)
    out = np.clip(arr * gains[None, None, :], 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8), mode="RGB")


def _tone_curve(im: Image.Image) -> Image.Image:
    # Gentle S-curve + slight highlight compression (natural)
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = arr
    out = np.clip((x - 0.5) * 1.18 + 0.5, 0, 1)
    out = out ** 0.985
    return Image.fromarray((out * 255).astype(np.uint8), mode="RGB")


def deterministic_pass(im: Image.Image) -> Image.Image:
    im = _gray_world_wb(im, strength=WB_STRENGTH)
    im = _tone_curve(im)

    im = ImageEnhance.Contrast(im).enhance(CONTRAST_GAIN)
    im = ImageEnhance.Color(im).enhance(COLOR_GAIN)
    im = ImageEnhance.Brightness(im).enhance(BRIGHTNESS_GAIN)
    im = ImageEnhance.Sharpness(im).enhance(SHARPNESS_GAIN)

    return im


# ---------------------------
# Edge-protect to stop warps
# ---------------------------

def _edge_mask(im: Image.Image, thresh: float) -> np.ndarray:
    g = np.asarray(im.convert("L")).astype(np.float32) / 255.0

    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5

    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-6)

    m = (mag > thresh).astype(np.float32)

    # soften (tiny 3x3 box blur)
    k = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.float32) / 9.0
    p = np.pad(m, ((1,1),(1,1)), mode="edge")
    m2 = np.empty_like(m)
    for y in range(m.shape[0]):
        for x in range(m.shape[1]):
            m2[y, x] = float((p[y:y+3, x:x+3] * k).sum())

    return np.clip(m2, 0, 1)


def protect_edges(base: Image.Image, ai: Image.Image, alpha: float, thresh: float) -> Image.Image:
    if ai.size != base.size:
        ai = ai.resize(base.size, Image.Resampling.LANCZOS)

    m = _edge_mask(base, thresh=thresh)
    m = np.clip(m * alpha, 0, 1)
    m3 = m[..., None]

    base_np = np.asarray(base).astype(np.float32)
    ai_np = np.asarray(ai).astype(np.float32)

    out = ai_np * (1.0 - m3) + base_np * m3
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def root():
    return {"ok": True}


@app.get("/version")
def version():
    return {
        "openai_version": getattr(openai, "__version__", "unknown"),
        "has_client_responses": hasattr(client, "responses"),
        "model": MAIN_MODEL,
        "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "heif_enabled": HEIF_ENABLED,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_pixels": MAX_PIXELS,
        "edge_protect_alpha": EDGE_PROTECT_ALPHA,
        "edge_thresh": EDGE_THRESH,
    }


@app.post("/enhance")
async def enhance(file: UploadFile = File(...), image_url: Optional[str] = None):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    if not hasattr(client, "responses"):
        raise HTTPException(500, "OpenAI SDK missing client.responses (check requirements + redeploy with Clear Cache).")

    # ---- Ingest ----
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")

    # Prevent Vercel 413: payload too large. :contentReference[oaicite:5]{index=5}
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large for Vercel Functions. Limit ~4.5MB. Your upload: {len(data)} bytes. "
                   "Production fix: upload to storage (S3/R2/Supabase) and send a URL instead.",
        )

    im = _load_rgb(data)
    im = _downscale_if_needed(im)

    # ---- Deterministic pro edits (guaranteed visible improvement) ----
    base = deterministic_pass(im)

    # ---- AI polish (forced edit) ----
    tool_size = _pick_tool_size(*base.size)
    data_url = _to_data_url_png(base)

    resp = client.responses.create(
        model=MAIN_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        tools=[{
            "type": "image_generation",
            "action": "edit",
            "input_fidelity": "high",
            "size": tool_size,
            "quality": "high",
        }],
    )

    calls = [o for o in resp.output if getattr(o, "type", None) == "image_generation_call"]
    if not calls:
        raise HTTPException(500, "No image_generation_call returned")

    call0 = calls[0]
    ai = _decode_b64_image(call0.result)

    # ---- Protect text/logos/wheels by restoring sharp edges from base ----
    final = protect_edges(base, ai, alpha=EDGE_PROTECT_ALPHA, thresh=EDGE_THRESH)

    out = io.BytesIO()
    final.save(out, format="PNG")

    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={
            "x-ai-used": "true",
            "x-tool-size": tool_size,
            "x-action": str(getattr(call0, "action", "unknown")),
            "x-heif": str(HEIF_ENABLED).lower(),
            "x-edge-protect-alpha": str(EDGE_PROTECT_ALPHA),
            "x-edge-thresh": str(EDGE_THRESH),
            "cache-control": "no-store",
        },
    )
