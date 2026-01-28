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

No local edits. If a change risks altering physical details, do not apply it.
Prefer minimal change over altering details. Photorealistic. No stylization.
""".strip()


def _safe_polish(img: Image.Image) -> Image.Image:
    """
    Very mild global polish to reduce "matte" WITHOUT halos/sloppy look.
    """
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Color(img).enhance(1.02)
    img = ImageEnhance.Sharpness(img).enhance(1.04)
    return img


def _resize_to_match(a: np.ndarray, b_img: Image.Image) -> np.ndarray:
    """
    Ensure b matches a's HxW for scoring.
    """
    h, w = a.shape[0], a.shape[1]
    if b_img.size != (w, h):
        b_img = b_img.resize((w, h), Image.LANCZOS)
    return np.array(b_img, dtype=np.uint8)


def _score_similarity(processed_np: np.ndarray, candidate_np: np.ndarray) -> float:
    """
    Lower score = closer to processed (less hallucination).
    Mix of pixel diff + edge diff (no OpenCV, no lockers).
    """
    a = processed_np.astype(np.int16)
    b = candidate_np.astype(np.int16)

    # Mean absolute pixel difference (RGB)
    pix = float(np.mean(np.abs(a - b)))

    # Edge-map difference (structure changes)
    a_edges = Image.fromarray(processed_np).convert("L").filter(ImageFilter.FIND_EDGES)
    b_edges = Image.fromarray(candidate_np).convert("L").filter(ImageFilter.FIND_EDGES)
    ae = np.array(a_edges, dtype=np.int16)
    be = np.array(b_edges, dtype=np.int16)
    edge = float(np.mean(np.abs(ae - be)))

    # Weighted total
    return (pix * 1.0) + (edge * 0.6)


def _call_edit(tmp_path: str) -> bytes:
    """
    One OpenAI edit call -> PNG bytes.
    """
    result = client.images.edit(
        model="gpt-image-1",
        image=open(tmp_path, "rb"),
        prompt=PROMPT,
        size="auto",
    )
    out_bytes = base64.b64decode(result.data[0].b64_json)
    return out_bytes


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # Keep detail but limit huge images
        MAX_SIZE = 2560
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # Deterministic base (your pipeline)
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # Save once; we will edit the same deterministic image twice
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # ---- BEST OF 2 ----
        out1 = _call_edit(tmp_path)
        out2 = _call_edit(tmp_path)

        img1 = Image.open(io.BytesIO(out1)).convert("RGB")
        img2 = Image.open(io.BytesIO(out2)).convert("RGB")

        cand1_np = _resize_to_match(processed_np, img1)
        cand2_np = _resize_to_match(processed_np, img2)

        s1 = _score_similarity(processed_np, cand1_np)
        s2 = _score_similarity(processed_np, cand2_np)

        chosen_img = img1 if s1 <= s2 else img2
        chosen_score = s1 if s1 <= s2 else s2

        # Mild polish (optional) to reduce matte look
        chosen_img = _safe_polish(chosen_img)

        buf = io.BytesIO()
        chosen_img.save(buf, format="PNG")

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-BO2-S1": str(s1),
                "X-BO2-S2": str(s2),
                "X-BO2-CHOSEN": "1" if s1 <= s2 else "2",
                "X-BO2-SCORE": str(chosen_score),
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
