import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageEnhance, ImageFilter
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image


app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"status": "ok"}


PROMPT = """
Enhance this exact photo for a car sales listing.

ABSOLUTE IMMUTABLE (DO NOT CHANGE ANY OF THESE):
- Wheels/rims/spokes/tires/center caps/center-cap logos (NO warping, NO blur, NO redraw).
- Badges and any branding/logos anywhere.
- Grille pattern/mesh/shape/texture.
- Headlights/taillights/DRL shapes and internal LED patterns.
- Window tint level.
- Materials/trim: do NOT add chrome, do NOT brighten blacked-out trim, do NOT change matte↔gloss.
- Any text, letters, numbers, icons, UI elements, screens, or buttons.
- Body shape, reflections structure, background objects/layout.

ALLOWED (GLOBAL PHOTO CORRECTIONS ONLY):
- Correct white balance / remove color cast
- Improve exposure + contrast naturally (not HDR)
- Recover highlights a bit
- Lift shadows slightly
- Mild noise reduction

No local edits. If any change risks altering details, DO NOT APPLY it.
Prefer minimal change over altering details. Photorealistic. No stylization.
""".strip()


def _safe_polish(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Color(img).enhance(1.02)
    img = ImageEnhance.Sharpness(img).enhance(1.04)
    return img


def _resize_to_match(a_np: np.ndarray, b_img: Image.Image) -> np.ndarray:
    h, w = a_np.shape[0], a_np.shape[1]
    if b_img.size != (w, h):
        b_img = b_img.resize((w, h), Image.LANCZOS)
    return np.array(b_img, dtype=np.uint8)


def _score_similarity(processed_np: np.ndarray, candidate_np: np.ndarray) -> float:
    a = processed_np.astype(np.int16)
    b = candidate_np.astype(np.int16)
    pix = float(np.mean(np.abs(a - b)))

    a_edges = Image.fromarray(processed_np).convert("L").filter(ImageFilter.FIND_EDGES)
    b_edges = Image.fromarray(candidate_np).convert("L").filter(ImageFilter.FIND_EDGES)
    ae = np.array(a_edges, dtype=np.int16)
    be = np.array(b_edges, dtype=np.int16)
    edge = float(np.mean(np.abs(ae - be)))

    return (pix * 1.0) + (edge * 0.7)


def _return_png(img: Image.Image, headers: dict) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


def _downscale_if_needed(img: Image.Image, max_dim: int = 4000) -> tuple[Image.Image, bool]:
    """
    If either dimension exceeds max_dim, downscale proportionally to fit within max_dim x max_dim.
    Returns (image, did_resize).
    """
    w, h = img.size
    if max(w, h) <= max_dim:
        return img, False

    # thumbnail keeps aspect ratio
    img = img.copy()
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img, True


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    raw = b""
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # ✅ NEW: auto-downscale huge uploads to max 4000px longest side
        original, resized = _downscale_if_needed(original, max_dim=4000)

        # deterministic base
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # OpenAI edit (single call)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        result = client.images.edit(
            model="gpt-image-1",
            image=open(tmp_path, "rb"),
            prompt=PROMPT,
            size="auto",
        )

        out_bytes = base64.b64decode(result.data[0].b64_json)
        ai_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")

        # score vs deterministic; if too different, reject
        ai_np = _resize_to_match(processed_np, ai_img)
        s = _score_similarity(processed_np, ai_np)

        # Hallucination gate
        SCORE_MAX = 16.0

        if s > SCORE_MAX:
            fallback = _safe_polish(processed)
            return _return_png(
                fallback,
                headers={
                    "X-AI-USED": "false",
                    "X-REASON": "diff_gate",
                    "X-SCORE": str(s),
                    "X-DOWNSCALED": "true" if resized else "false",
                    "X-FINAL-WH": f"{original.size[0]}x{original.size[1]}",
                },
            )

        final = _safe_polish(ai_img)
        return _return_png(
            final,
            headers={
                "X-AI-USED": "true",
                "X-SCORE": str(s),
                "X-DOWNSCALED": "true" if resized else "false",
                "X-FINAL-WH": f"{original.size[0]}x{original.size[1]}",
            },
        )

    except Exception as e:
        msg = str(e)
        if "billing_hard_limit" in msg or "hard limit" in msg or "billing" in msg:
            # deterministic fallback if billing blocks AI
            try:
                # if original is already loaded, use it; otherwise decode from raw
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                img, resized = _downscale_if_needed(img, max_dim=4000)
                processed_np = enhance_image(img).astype(np.uint8)
                processed = Image.fromarray(processed_np, mode="RGB")
                fallback = _safe_polish(processed)
                return _return_png(
                    fallback,
                    headers={
                        "X-AI-USED": "false",
                        "X-REASON": "billing",
                        "X-DOWNSCALED": "true" if resized else "false",
                        "X-FINAL-WH": f"{img.size[0]}x{img.size[1]}",
                    },
                )
            except Exception:
                pass

        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
