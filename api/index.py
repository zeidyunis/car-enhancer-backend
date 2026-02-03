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


# Stronger: instruct "global transform only"
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

ALLOWED (GLOBAL ONLY — must apply uniformly to the whole image, not locally):
- Correct white balance / remove cast
- Small exposure + contrast improvement (natural)
- Mild highlight recovery, mild shadow lift
- Mild noise reduction only if needed (do not smear texture)

BANNED:
- No object edits, no additions, no removals, no “improvements” that change details.
- No local retouching, no HDR/clarity look, no sharpening halos, no fake reflections.
- Do not re-draw the car or background. Keep every edge and feature identical.

Photorealistic.
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
    fitted = ImageOps.contain(img, (canvas_w, canvas_h), method=Image.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    x = (canvas_w - fitted.size[0]) // 2
    y = (canvas_h - fitted.size[1]) // 2
    canvas.paste(fitted, (x, y))
    box = (x, y, x + fitted.size[0], y + fitted.size[1])
    return canvas, box


def _write_temp_png(pil_img: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    pil_img.save(tmp_path, format="PNG", optimize=True)
    return tmp_path


def _mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    """
    Simple deviation guard: mean absolute pixel difference in [0..255].
    Works as a blunt hammer to catch major hallucinations.
    """
    a_np = np.asarray(a).astype(np.int16)
    b_np = np.asarray(b).astype(np.int16)
    if a_np.shape != b_np.shape:
        return 9999.0
    return float(np.mean(np.abs(a_np - b_np)))


def _call_ai_edit(det_path: str, orig_path: str, size_str: str, out_fmt: str) -> Image.Image:
    """
    Use /v1/images/edits with:
    - multiple images (det + original) to anchor the edit
    - gpt-image-1 with input_fidelity="high" (stronger preservation)
    - explicit quality/output_format/output_compression
    """
    # Map our output format to API output_format
    api_out_fmt = "png" if out_fmt == "PNG" else "jpeg"

    result = client.images.edit(
        model="gpt-image-1",
        image=[open(det_path, "rb"), open(orig_path, "rb")],
        prompt=PROMPT,
        size=size_str,
        input_fidelity="high",         # only supported for gpt-image-1
        quality="medium",              # avoid "auto"
        output_format=api_out_fmt,     # png/jpeg/webp supported for GPT image models
        output_compression=95          # only applies for jpeg/webp; safe to send
    )

    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    tmp_paths: list[str] = []
    try:
        raw = await file.read()

        pil_in = Image.open(io.BytesIO(raw))
        pil_in = ImageOps.exif_transpose(pil_in)
        original_full = pil_in.convert("RGB")

        orig_w, orig_h = original_full.size
        out_fmt, out_mime, out_ext = _guess_output_format(file, pil_in.format)

        # 1) pick API canvas size based on original orientation
        canvas_w, canvas_h = _choose_api_canvas(orig_w, orig_h)
        size_str = f"{canvas_w}x{canvas_h}"

        # 2) pad to canvas (no crop), keep box of real content
        canvas_img, box = _pad_to_canvas(original_full, canvas_w, canvas_h)

        # 3) deterministic on canvas
        det_np = enhance_image(canvas_img).astype(np.uint8)
        det_img = Image.fromarray(det_np, mode="RGB")

        # Write temp images for API
        det_path = _write_temp_png(det_img)
        orig_path = _write_temp_png(canvas_img)
        tmp_paths.extend([det_path, orig_path])

        # 4) AI edit on deterministic canvas (forced size, high input fidelity)
        ai_canvas = _call_ai_edit(det_path, orig_path, size_str=size_str, out_fmt=out_fmt)

        # 5) crop back to ONLY real content (removes borders)
        ai_cropped = ai_canvas.crop(box)
        det_cropped = det_img.crop(box)

        # 6) resize back to original uploaded size (exact)
        final_ai = ai_cropped.resize((orig_w, orig_h), Image.LANCZOS)
        final_det = det_cropped.resize((orig_w, orig_h), Image.LANCZOS)

        # 7) Deviation guard:
        # If AI output deviates too much, return deterministic instead.
        # Tune threshold based on your dataset (start conservative).
        diff = _mean_abs_diff(final_ai, final_det)
        MAX_DIFF = float(os.getenv("MAX_AI_DIFF", "6.5"))  # lower = stricter

        if diff > MAX_DIFF:
            final = final_det
            used_ai = "false"
            reason = f"diff_guard_triggered:{diff:.2f}>{MAX_DIFF:.2f}"
        else:
            final = final_ai
            used_ai = "true"
            reason = f"diff_ok:{diff:.2f}<={MAX_DIFF:.2f}"

        body = _save_bytes(final, out_fmt)

        return Response(
            content=body,
            media_type=out_mime,
            headers={
                "X-AI-USED": used_ai,
                "X-AI-REASON": reason,
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

    finally:
        # cleanup temp files
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
