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


def _guess_output_format(upload: UploadFile, pil_format: str | None) -> tuple[str, str]:
    """
    Returns: (PIL_format, mime_type)
    We only serve JPEG or PNG to keep behavior predictable.
    """
    ct = (upload.content_type or "").lower()
    pf = (pil_format or "").upper()

    # prefer explicit content-type
    if "png" in ct or pf == "PNG":
        return "PNG", "image/png"
    if "jpeg" in ct or "jpg" in ct or pf in ("JPEG", "JPG"):
        return "JPEG", "image/jpeg"

    # default for HEIC/HEIF/WEBP/etc.
    return "JPEG", "image/jpeg"


def _save_image_bytes(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "PNG":
        img.save(buf, format="PNG", optimize=True)
    else:
        # high quality, stable output (no matte)
        img.save(buf, format="JPEG", quality=92, subsampling=1, optimize=True)
    return buf.getvalue()


def _fit_to_exact_canvas_no_crop(img: Image.Image, ref_canvas: Image.Image) -> Image.Image:
    """
    Force output to EXACT ref_canvas size with NO cropping.
    If aspect differs, we pad/letterbox using a blurred reference background.
    """
    out_w, out_h = ref_canvas.size

    # background = blurred original (same size)
    bg = ref_canvas.copy().filter(ImageFilter.GaussianBlur(radius=18))

    # fit img inside without cropping
    fitted = ImageOps.contain(img, (out_w, out_h), method=Image.LANCZOS)

    x = (out_w - fitted.size[0]) // 2
    y = (out_h - fitted.size[1]) // 2
    bg.paste(fitted, (x, y))

    return bg


def _call_ai_edit(tmp_path: str) -> Image.Image:
    result = client.images.edit(
        model="gpt-image-1.5",
        image=open(tmp_path, "rb"),
        prompt=PROMPT,
        size="auto",
    )
    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()

        # IMPORTANT: keep original orientation correct (iPhone EXIF)
        pil_in = Image.open(io.BytesIO(raw))
        pil_in = ImageOps.exif_transpose(pil_in)
        original_full = pil_in.convert("RGB")

        # Lock final output size to EXACT uploaded size
        out_size = original_full.size

        # Decide output format based on upload
        out_fmt, out_mime = _guess_output_format(file, pil_in.format)

        # Working copy for deterministic + AI (cost control)
        working = original_full.copy()
        MAX_WORK_DIM = 2560
        if max(working.size) > MAX_WORK_DIM:
            working.thumbnail((MAX_WORK_DIM, MAX_WORK_DIM), Image.LANCZOS)

        # Deterministic step
        processed_np = enhance_image(working).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # AI edits the deterministic image (force edit workflow)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        ai_img = _call_ai_edit(tmp_path)

        # NOW: force AI output back onto exact original canvas (no crop)
        ref_canvas = original_full.resize(out_size, Image.LANCZOS)
        final = _fit_to_exact_canvas_no_crop(ai_img, ref_canvas)

        body = _save_image_bytes(final, out_fmt)

        return Response(
            content=body,
            media_type=out_mime,
            headers={
                "X-AI-USED": "true",
                "X-OUT-SIZE": f"{out_size[0]}x{out_size[1]}",
                "X-OUT-FORMAT": out_fmt,
                "X-WORK-SIZE": f"{working.size[0]}x{working.size[1]}",
                # inline prevents “download python file” weirdness in browsers
                "Content-Disposition": f'inline; filename="enhanced.{out_fmt.lower() if out_fmt != "JPEG" else "jpg"}"',
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
