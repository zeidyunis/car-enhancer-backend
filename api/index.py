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


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original_full = Image.open(io.BytesIO(raw)).convert("RGB")

        # ✅ Final output will ALWAYS be same size as uploaded
        out_size = original_full.size

        # Working copy for processing/AI to keep cost/time stable
        working = original_full.copy()
        MAX_WORK_DIM = 2560
        if max(working.size) > MAX_WORK_DIM:
            working.thumbnail((MAX_WORK_DIM, MAX_WORK_DIM), Image.LANCZOS)

        # Deterministic base
        processed_np = enhance_image(working).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # Save for AI edit
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # ✅ AI edit (no diff gate; AI is always used)
        ai_img = _call_ai_edit(tmp_path)

        # Return exact uploaded size, no crop
        canvas_bg = original_full.resize(out_size, Image.LANCZOS)
        final = _fit_into_canvas_no_crop(ai_img, canvas_bg, out_size)

        return _return_png(
            final,
            headers={
                "X-AI-USED": "true",
                "X-OUT-SIZE": f"{out_size[0]}x{out_size[1]}",
                "X-WORK-SIZE": f"{working.size[0]}x{working.size[1]}",
                "X-AI-SIZE": f"{ai_img.size[0]}x{ai_img.size[1]}",
            },
        )

    except Exception as e:
        # ✅ No silent fallback. If OpenAI fails, you SEE it.
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
