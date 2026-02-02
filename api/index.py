import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageOps
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


ALLOWED_SIZES = [(1024, 1024), (1536, 1024), (1024, 1536)]  # w,h


def _guess_output_format(upload: UploadFile, pil_format: str | None) -> tuple[str, str, str]:
    ct = (upload.content_type or "").lower()
    pf = (pil_format or "").upper()
    if "png" in ct or pf == "PNG":
        return "PNG", "image/png", "png"
    if "jpeg" in ct or "jpg" in ct or pf in ("JPEG", "JPG"):
        return "JPEG", "image/jpeg", "jpg"
    return "JPEG", "image/jpeg", "jpg"


def _save_bytes(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "PNG":
        img.save(buf, format="PNG", optimize=True)
    else:
        img.save(buf, format="JPEG", quality=92, subsampling=1, optimize=True)
    return buf.getvalue()


def _choose_api_canvas(w: int, h: int) -> tuple[int, int]:
    ar = w / h
    if ar > 1.15:
        return 1536, 1024
    if ar < 0.87:
        return 1024, 1536
    return 1024, 1024


def _pad_to_canvas(img: Image.Image, canvas_w: int, canvas_h: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Fit original image into allowed API canvas WITHOUT cropping.
    Returns:
      - canvas image (RGB) of size (canvas_w, canvas_h)
      - content box (left, top, right, bottom) where real pixels live
    """
    # resize to fit inside canvas
    fitted = ImageOps.contain(img, (canvas_w, canvas_h), method=Image.LANCZOS)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))  # dark neutral border
    x = (canvas_w - fitted.size[0]) // 2
    y = (canvas_h - fitted.size[1]) // 2
    canvas.paste(fitted, (x, y))

    box = (x, y, x + fitted.size[0], y + fitted.size[1])
    return canvas, box


def _call_ai_edit(tmp_path: str, size_str: str) -> Image.Image:
    result = client.images.edit(
        model="gpt-image-1",
        image=open(tmp_path, "rb"),
        prompt=PROMPT,
        size=size_str,  # force one of allowed sizes; prevents "auto" surprises
    )
    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()

        pil_in = Image.open(io.BytesIO(raw))
        pil_in = ImageOps.exif_transpose(pil_in)  # iPhone rotation fix
        original_full = pil_in.convert("RGB")

        orig_w, orig_h = original_full.size
        out_fmt, out_mime, out_ext = _guess_output_format(file, pil_in.format)

        # 1) pick API canvas size based on original orientation
        canvas_w, canvas_h = _choose_api_canvas(orig_w, orig_h)
        size_str = f"{canvas_w}x{canvas_h}"

        # 2) pad to canvas (no crop), keep box of real content
        canvas_img, box = _pad_to_canvas(original_full, canvas_w, canvas_h)

        # 3) deterministic on canvas (so the AI sees same canvas)
        det_np = enhance_image(canvas_img).astype(np.uint8)
        det_img = Image.fromarray(det_np, mode="RGB")

        # 4) AI edit on deterministic canvas (forced size)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        det_img.save(tmp_path)

        ai_canvas = _call_ai_edit(tmp_path, size_str=size_str)

        # 5) crop back to ONLY real content (removes borders)
        ai_cropped = ai_canvas.crop(box)

        # 6) resize back to original uploaded size (exact)
        final = ai_cropped.resize((orig_w, orig_h), Image.LANCZOS)

        body = _save_bytes(final, out_fmt)

        return Response(
            content=body,
            media_type=out_mime,
            headers={
                "X-AI-USED": "true",
                "X-ORIG": f"{orig_w}x{orig_h}",
                "X-API-CANVAS": size_str,
                "X-BOX": f"{box[0]},{box[1]},{box[2]},{box[3]}",
                "X-OUT-FORMAT": out_fmt,
                "Content-Disposition": f'inline; filename="enhanced.{out_ext}"',
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
