import io
import os
import base64
import tempfile
import traceback

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

FRAMING:
- Keep the original framing/composition. Do NOT crop, zoom, rotate, or change aspect ratio.

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


def _score_similarity(processed: Image.Image, candidate: Image.Image) -> float:
    # downscaled grayscale score (less false rejections)
    a = processed.convert("L")
    b = candidate.convert("L")

    a_small = a.copy()
    a_small.thumbnail((512, 512), Image.LANCZOS)
    b_small = b.resize(a_small.size, Image.LANCZOS)

    a_np = np.array(a_small, dtype=np.int16)
    b_np = np.array(b_small, dtype=np.int16)
    pix = float(np.mean(np.abs(a_np - b_np)))

    a_e = a_small.filter(ImageFilter.FIND_EDGES)
    b_e = b_small.filter(ImageFilter.FIND_EDGES)
    ae = np.array(a_e, dtype=np.int16)
    be = np.array(b_e, dtype=np.int16)
    edge = float(np.mean(np.abs(ae - be)))

    return (pix * 1.0) + (edge * 0.25)


def _fit_into_canvas_no_crop(img: Image.Image, canvas_bg: Image.Image, out_size: tuple[int, int]) -> Image.Image:
    """
    Force EXACT output dimensions (out_size) with NO cropping.
    Letterbox/pad if aspect differs.
    """
    out_w, out_h = out_size

    # Background: blurred original (already at out_size)
    bg = canvas_bg.copy().filter(ImageFilter.GaussianBlur(radius=18))

    # Fit candidate inside output size without cropping
    fitted = ImageOps.contain(img, (out_w, out_h), method=Image.LANCZOS)

    x = (out_w - fitted.size[0]) // 2
    y = (out_h - fitted.size[1]) // 2
    bg.paste(fitted, (x, y))

    return bg


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    raw = b""
    try:
        raw = await file.read()
        original_full = Image.open(io.BytesIO(raw)).convert("RGB")

        # ✅ Keep the original uploaded dimensions as the final output target
        out_size = original_full.size

        # Make a working copy for processing/AI to avoid huge costs
        working = original_full.copy()
        MAX_WORK_DIM = 2560
        if max(working.size) > MAX_WORK_DIM:
            working.thumbnail((MAX_WORK_DIM, MAX_WORK_DIM), Image.LANCZOS)

        # Deterministic base on working
        processed_np = enhance_image(working).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # AI edit on deterministic image
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        ai_img = _call_ai_edit(tmp_path)

        # Diff-gate against deterministic (same working size)
        s = _score_similarity(processed, ai_img)
        SCORE_MAX = 30.0  # adjust if needed

        # Prepare background at FINAL output size
        canvas_bg = original_full.resize(out_size, Image.LANCZOS)

        if s > SCORE_MAX:
            # Reject AI -> deterministic, but still return EXACT original W×H
            safe = _fit_into_canvas_no_crop(processed, canvas_bg, out_size)
            return _return_png(
                safe,
                headers={
                    "X-AI-USED": "false",
                    "X-REASON": "diff_gate",
                    "X-SCORE": str(s),
                    "X-OUT-SIZE": f"{out_size[0]}x{out_size[1]}",
                    "X-WORK-SIZE": f"{working.size[0]}x{working.size[1]}",
                },
            )

        # Accept AI -> still return EXACT original W×H
        final = _fit_into_canvas_no_crop(ai_img, canvas_bg, out_size)
        return _return_png(
            final,
            headers={
                "X-AI-USED": "true",
                "X-SCORE": str(s),
                "X-OUT-SIZE": f"{out_size[0]}x{out_size[1]}",
                "X-WORK-SIZE": f"{working.size[0]}x{working.size[1]}",
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
