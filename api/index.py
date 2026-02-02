import io
import os
import base64
import tempfile
import traceback
import math

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageFilter, ImageOps
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image


app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"status": "ok"}


PROMPT = """
Edit (not recreate) this exact photo for a car sales listing.

FRAMING (STRICT):
- Keep the original framing/composition exactly the same.
- Do NOT crop, zoom, rotate, or change aspect ratio.

ABSOLUTE IMMUTABLE (DO NOT CHANGE):
- Wheels/rims/spokes/tires/center caps/center-cap logos
- Badges/logos anywhere
- Grille pattern/mesh/shape/texture
- Headlights/taillights/DRL shapes and internal patterns
- Any text/numbers/icons/screens/buttons
- Body shape, reflections structure, background objects/layout
- Materials/trim: do NOT add chrome/gloss; do NOT brighten blacked-out trim; do NOT change matte↔gloss

ALLOWED (GLOBAL ONLY):
- Correct white balance / remove cast
- Small exposure + contrast improvement (natural)
- Mild highlight recovery, mild shadow lift
- Mild noise reduction only if needed (do not smear texture)

No local retouching, no HDR/clarity look. Photorealistic.
""".strip()


def _return_png(img: Image.Image, headers: dict) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


def _call_ai_edit(tmp_path: str) -> Image.Image:
    result = client.images.edit(
        model="gpt-image-1.5",
        image=open(tmp_path, "rb"),
        prompt=PROMPT,
        size="auto",
    )
    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


def _fit_into_canvas_no_crop(img: Image.Image, canvas_bg: Image.Image, out_size: tuple[int, int]) -> Image.Image:
    """
    Force EXACT output dimensions (out_size) with NO cropping.
    If aspect differs, it pads (letterbox) over a blurred background.
    """
    out_w, out_h = out_size
    bg = canvas_bg.copy().filter(ImageFilter.GaussianBlur(radius=18))

    fitted = ImageOps.contain(img, (out_w, out_h), method=Image.LANCZOS)

    x = (out_w - fitted.size[0]) // 2
    y = (out_h - fitted.size[1]) // 2
    bg.paste(fitted, (x, y))

    return bg


def _edge_map(img: Image.Image, size=(256, 256)) -> np.ndarray:
    """
    Edge map for framing comparison, robust to color/exposure changes.
    """
    g = img.convert("L").resize(size, Image.LANCZOS)
    e = g.filter(ImageFilter.FIND_EDGES)
    arr = np.array(e, dtype=np.float32)
    # normalize
    arr -= arr.mean()
    std = float(arr.std()) + 1e-6
    arr /= std
    return arr


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between two same-shaped arrays.
    """
    a = a.reshape(-1)
    b = b.reshape(-1)
    # both are already normalized, so corr is mean(a*b)
    return float(np.mean(a * b))


def _framing_gate(processed: Image.Image, ai_img: Image.Image) -> tuple[bool, dict]:
    """
    Returns (ok, info_dict)
    Only rejects when AI changed framing (crop/zoom/shift/aspect).
    """
    pw, ph = processed.size
    aw, ah = ai_img.size

    # 1) Aspect ratio tolerance (log-space)
    ar_p = pw / ph
    ar_a = aw / ah
    ar_delta = abs(math.log(ar_a / ar_p))  # 0 == identical
    AR_TOL = 0.02  # ~2% tolerance

    if ar_delta > AR_TOL:
        return False, {"gate": "aspect_ratio", "ar_delta": ar_delta, "ar_tol": AR_TOL}

    # 2) Edge-layout correlation (detect zoom/crop/shift)
    p_edges = _edge_map(processed)
    # Compare to AI resized to processed size (no crop)
    ai_rs = ai_img.resize((pw, ph), Image.LANCZOS)
    a_edges = _edge_map(ai_rs)

    corr = _corrcoef(p_edges, a_edges)

    # High threshold: we only want to reject obvious framing changes.
    CORR_MIN = 0.80
    if corr < CORR_MIN:
        return False, {"gate": "edge_corr", "corr": corr, "corr_min": CORR_MIN}

    return True, {"gate": "ok", "ar_delta": ar_delta, "corr": corr}


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original_full = Image.open(io.BytesIO(raw)).convert("RGB")

        # Final output must match uploaded dimensions exactly
        out_size = original_full.size

        # Working copy for processing/AI
        working = original_full.copy()
        MAX_WORK_DIM = 2560
        if max(working.size) > MAX_WORK_DIM:
            working.thumbnail((MAX_WORK_DIM, MAX_WORK_DIM), Image.LANCZOS)

        # Deterministic base
        processed_np = enhance_image(working).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # Save deterministic image for AI edit
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # AI edit
        ai_img = _call_ai_edit(tmp_path)

        # ✅ Framing-only gate (no pixel/quality/hallucination gate)
        ok, info = _framing_gate(processed, ai_img)

        canvas_bg = original_full.resize(out_size, Image.LANCZOS)

        if not ok:
            # Reject AI only when it changes framing
            safe = _fit_into_canvas_no_crop(processed, canvas_bg, out_size)
            return _return_png(
                safe,
                headers={
                    "X-AI-USED": "false",
                    "X-REASON": "framing_gate",
                    "X-GATE": info.get("gate", ""),
                    "X-AR-DELTA": str(info.get("ar_delta", "")),
                    "X-CORR": str(info.get("corr", "")),
                    "X-OUT-SIZE": f"{out_size[0]}x{out_size[1]}",
                    "X-WORK-SIZE": f"{working.size[0]}x{working.size[1]}",
                },
            )

        # Accept AI
        final = _fit_into_canvas_no_crop(ai_img, canvas_bg, out_size)
        return _return_png(
            final,
            headers={
                "X-AI-USED": "true",
                "X-GATE": "ok",
                "X-AR-DELTA": str(info.get("ar_delta", "")),
                "X-CORR": str(info.get("corr", "")),
                "X-OUT-SIZE": f"{out_size[0]}x{out_size[1]}",
                "X-WORK-SIZE": f"{working.size[0]}x{working.size[1]}",
                "X-AI-SIZE": f"{ai_img.size[0]}x{ai_img.size[1]}",
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
