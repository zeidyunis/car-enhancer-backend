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
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))  # dark neutral border
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


def _to_gray_u8(img: Image.Image, width: int = 512) -> np.ndarray:
    """
    Downscale (for speed) and convert to grayscale uint8.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    scale = min(1.0, width / float(w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.asarray(small).astype(np.float32)
    # luminance
    gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def _edge_map(gray_u8: np.ndarray) -> np.ndarray:
    """
    Simple gradient magnitude edge map (no OpenCV needed).
    Returns float32 in [0..1].
    """
    g = gray_u8.astype(np.float32) / 255.0
    # Sobel-ish central diffs
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    mag = np.sqrt(gx * gx + gy * gy)
    # normalize robustly
    p = np.percentile(mag, 99.0) if mag.size else 1.0
    if p <= 1e-6:
        return np.zeros_like(mag, dtype=np.float32)
    return np.clip(mag / p, 0.0, 1.0).astype(np.float32)


def _edge_similarity(a_img: Image.Image, b_img: Image.Image) -> float:
    """
    Compare structural similarity via edge maps:
      similarity = 1 - normalized MAE(edgeA, edgeB)
    Higher = more structurally identical.
    """
    a_g = _to_gray_u8(a_img, width=int(os.getenv("GUARD_DOWNSCALE", "512")))
    b_g = _to_gray_u8(b_img, width=a_g.shape[1] if a_g.ndim == 2 else 512)
    if a_g.shape != b_g.shape:
        # best effort: resize b to a
        b_img2 = b_img.resize((a_img.size[0], a_img.size[1]), Image.BILINEAR)
        b_g = _to_gray_u8(b_img2, width=int(os.getenv("GUARD_DOWNSCALE", "512")))
    ea = _edge_map(a_g)
    eb = _edge_map(b_g)
    if ea.shape != eb.shape or ea.size == 0:
        return 0.0
    mae = float(np.mean(np.abs(ea - eb)))  # 0..1
    sim = 1.0 - mae
    return max(0.0, min(1.0, sim))


def _call_ai_edit(det_path: str, orig_path: str, size_str: str, out_fmt: str) -> Image.Image:
    """
    Use /v1/images/edits with strong input anchoring:
    - multiple images (deterministic + original) to anchor
    - gpt-image-1 with input_fidelity="high"
    - explicit quality + output_format + output_compression
    """
    api_out_fmt = "png" if out_fmt == "PNG" else "jpeg"

    # Use conservative quality by default; you can override via env
    quality = os.getenv("AI_QUALITY", "low")  # low|medium|high|auto
    compression = int(os.getenv("AI_OUTPUT_COMPRESSION", "95"))  # 0-100
    fidelity = os.getenv("AI_INPUT_FIDELITY", "high")  # for gpt-image-1: low|high

    with open(det_path, "rb") as f_det, open(orig_path, "rb") as f_orig:
        result = client.images.edit(
            model="gpt-image-1",
            image=[f_det, f_orig],
            prompt=PROMPT,
            size=size_str,
            input_fidelity=fidelity,
            quality=quality,
            output_format=api_out_fmt,
            output_compression=compression,
        )

    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    tmp_paths: list[str] = []
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
        det_path = _write_temp_png(det_img)
        orig_path = _write_temp_png(canvas_img)
        tmp_paths.extend([det_path, orig_path])

        # Allow turning AI off entirely via env
        if os.getenv("DISABLE_AI", "0") == "1":
            ai_canvas = None
            ai_error = "disabled_by_env"
        else:
            ai_canvas = _call_ai_edit(det_path, orig_path, size_str=size_str, out_fmt=out_fmt)
            ai_error = ""

        # 5) crop back to ONLY real content (removes borders)
        det_cropped = det_img.crop(box)
        det_final = det_cropped.resize((orig_w, orig_h), Image.LANCZOS)

        # If AI failed/disabled, return deterministic
        if ai_canvas is None:
            body = _save_bytes(det_final, out_fmt)
            return Response(
                content=body,
                media_type=out_mime,
                headers={
                    "X-AI-USED": "false",
                    "X-AI-REASON": ai_error or "ai_not_run",
                    "X-ORIG": f"{orig_w}x{orig_h}",
                    "X-API-CANVAS": size_str,
                    "X-BOX": f"{box[0]},{box[1]},{box[2]},{box[3]}",
                    "X-OUT-FORMAT": out_fmt,
                    "Content-Disposition": f'inline; filename="enhanced.{out_ext}"',
                },
            )

        ai_cropped = ai_canvas.crop(box)
        ai_final = ai_cropped.resize((orig_w, orig_h), Image.LANCZOS)

        # 6) STRUCTURE GUARD (edge similarity)
        # This is much better than raw pixel diff for your use-case:
        # - allows global WB/exposure changes
        # - catches "car model changed / wheels changed / grille changed" because edges move
        edge_sim = _edge_similarity(ai_final, det_final)

        # Threshold: higher is stricter. Start at 0.965 and tune.
        MIN_EDGE_SIM = float(os.getenv("MIN_EDGE_SIM", "0.965"))

        if edge_sim < MIN_EDGE_SIM:
            final = det_final
            used_ai = "false"
            reason = f"edge_guard_triggered:{edge_sim:.4f}<{MIN_EDGE_SIM:.4f}"
        else:
            final = ai_final
            used_ai = "true"
            reason = f"edge_ok:{edge_sim:.4f}>={MIN_EDGE_SIM:.4f}"

        body = _save_bytes(final, out_fmt)

        return Response(
            content=body,
            media_type=out_mime,
            headers={
                "X-AI-USED": used_ai,
                "X-AI-REASON": reason,
                "X-EDGE-SIM": f"{edge_sim:.4f}",
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
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
