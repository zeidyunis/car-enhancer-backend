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

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAIN_MODEL = os.getenv("MAIN_MODEL", "gpt-5.2")
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(4000 * 4000)))  # 16MP cap

# Composite strength for restoring original edges/text/logos (0..1)
EDGE_PROTECT_ALPHA = float(os.getenv("EDGE_PROTECT_ALPHA", "0.85"))
# Edge detection threshold (lower = protect more details)
EDGE_THRESH = float(os.getenv("EDGE_THRESH", "0.22"))


PROMPT = """
You MUST edit the provided image (NOT generate a new one).
Goal: professional listing-quality photo.

DO NOT CHANGE (IMMUTABLE):
- Wheels/rims/tires/spokes/center caps and their logos
- Any badges/emblems/logos
- Grille design/pattern/shape
- Headlights/taillights/DRL shapes and inner patterns
- Any text/numbers/icons/screens/buttons (must remain perfectly readable, not warped)
- No new chrome/trim, no repaint, no geometry changes, no object add/remove
- Keep framing identical (no crop/zoom/rotate)

ALLOWED (GLOBAL ONLY):
- Correct white balance / remove color cast
- Improve exposure and contrast naturally
- Recover highlights mildly, lift shadows mildly
- Increase clarity/sharpness subtly (no halos)
- Reduce noise subtly (do not smear texture)
Photorealistic.
""".strip()


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc(), "path": str(request.url)},
    )


def load_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def downscale_if_needed(im: Image.Image) -> Image.Image:
    w, h = im.size
    if (w * h) <= MAX_PIXELS:
        return im
    scale = (MAX_PIXELS / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def pick_tool_size(w: int, h: int) -> str:
    if w > h:
        return "1536x1024"
    if h > w:
        return "1024x1536"
    return "1024x1024"


def to_data_url_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ---------------------------
# Deterministic “pro” edits
# ---------------------------

def gray_world_white_balance(im: Image.Image, strength: float = 0.9) -> Image.Image:
    """
    Gray-world WB (safe). strength 0..1 (higher = stronger correction)
    """
    arr = np.asarray(im).astype(np.float32) / 255.0
    means = arr.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = float(means.mean())
    gains = gray / means
    gains = 1.0 + strength * (gains - 1.0)
    out = np.clip(arr * gains[None, None, :], 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8), mode="RGB")


def tone_curve(im: Image.Image) -> Image.Image:
    """
    Mild S-curve, keeps it natural.
    """
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = arr
    # simple smooth S curve
    out = np.clip((x - 0.5) * 1.15 + 0.5, 0, 1)
    # tiny highlight compression
    out = out ** 0.98
    return Image.fromarray((out * 255).astype(np.uint8), mode="RGB")


def pro_color_pass(im: Image.Image) -> Image.Image:
    im = gray_world_white_balance(im, strength=0.9)
    im = tone_curve(im)

    # slightly more contrast, slightly more color, slight brightness
    im = ImageEnhance.Contrast(im).enhance(1.08)
    im = ImageEnhance.Color(im).enhance(1.06)
    im = ImageEnhance.Brightness(im).enhance(1.02)

    # subtle sharpness (avoid halos)
    im = ImageEnhance.Sharpness(im).enhance(1.12)
    return im


# ---------------------------
# Edge-protect composite
# ---------------------------

def edge_mask(im: Image.Image, thresh: float = 0.22) -> np.ndarray:
    """
    Build a mask of strong edges (logos/text often live here).
    Returns float mask 0..1, shape (H,W).
    """
    g = np.asarray(im.convert("L")).astype(np.float32) / 255.0

    # simple Sobel gradients (no opencv)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)

    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5

    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-6)

    # threshold -> mask
    m = (mag > thresh).astype(np.float32)

    # soften mask edges a little using a tiny blur via convolution
    # (manual 3x3 box blur)
    k = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.float32) / 9.0
    m2 = m.copy()
    # pad
    p = np.pad(m, ((1,1),(1,1)), mode="edge")
    for y in range(m.shape[0]):
        for x in range(m.shape[1]):
            m2[y, x] = float((p[y:y+3, x:x+3] * k).sum())

    # normalize 0..1
    m2 = np.clip(m2, 0, 1)
    return m2


def composite_protect_edges(original: Image.Image, ai: Image.Image, alpha: float, thresh: float) -> Image.Image:
    """
    Put original pixels back on strong edges to prevent warped text/logos.
    alpha controls how strongly we restore (0..1).
    """
    if ai.size != original.size:
        ai = ai.resize(original.size, Image.Resampling.LANCZOS)

    m = edge_mask(original, thresh=thresh)  # 0..1
    m = np.clip(m * alpha, 0, 1)           # scale by alpha

    orig_np = np.asarray(original).astype(np.float32)
    ai_np = np.asarray(ai).astype(np.float32)

    m3 = m[..., None]  # H,W,1

    out = ai_np * (1.0 - m3) + orig_np * m3
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

    original = load_image(data)
    safe = downscale_if_needed(original)

    # 1) Deterministic pro pass (guaranteed improvement)
    pre = pro_color_pass(safe)

    tool_size = pick_tool_size(*pre.size)
    image_url = to_data_url_png(pre)

    if not hasattr(client, "responses"):
        raise HTTPException(500, "OpenAI SDK missing client.responses")

    # 2) AI polish (light)
    resp = client.responses.create(
        model=MAIN_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        tools=[
            {
                "type": "image_generation",
                "action": "edit",
                "input_fidelity": "high",
                "size": tool_size,
                "quality": "high",
            }
        ],
    )

    calls = [o for o in resp.output if getattr(o, "type", None) == "image_generation_call"]
    if not calls:
        raise HTTPException(500, "No image_generation_call returned")

    ai = decode_image(calls[0].result)

    # 3) Anti-warp: restore original edges/text/logos
    # Use "pre" as the base “original” for compositing so colors match,
    # but preserve the sharp edge pixels from pre (not from raw upload).
    fixed = composite_protect_edges(pre, ai, alpha=EDGE_PROTECT_ALPHA, thresh=EDGE_THRESH)

    out = io.BytesIO()
    fixed.save(out, format="PNG")

    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={
            "x-ai-used": "true",
            "x-tool-size": tool_size,
            "x-returned-size": f"{fixed.size[0]}x{fixed.size[1]}",
            "x-edge-protect-alpha": str(EDGE_PROTECT_ALPHA),
            "x-edge-thresh": str(EDGE_THRESH),
            "cache-control": "no-store",
        },
    )
