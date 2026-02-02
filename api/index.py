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

IMPORTANT FRAMING RULE:
- Keep the original framing and composition exactly the same.
- Do NOT crop, zoom, rotate, or change aspect ratio.

ABSOLUTE IMMUTABLE (DO NOT CHANGE):
- Wheels/rims/spokes/tires/center caps/center-cap logos (no warping, no blur, no redraw)
- Badges/logos anywhere
- Grille pattern/mesh/shape/texture
- Headlights/taillights/DRL shapes and internal LED patterns
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
        model="gpt-image-1",
        image=open(tmp_path, "rb"),
        prompt=PROMPT,
        size="auto",
    )
    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


def _match_to_reference_no_crop(img: Image.Image, reference: Image.Image) -> Image.Image:
    """
    Force output to reference size WITHOUT cropping.
    If aspect differs, we letterbox using a blurred reference background.
    """
    ref_w, ref_h = reference.size
    fitted = ImageOps.contain(img, (ref_w, ref_h), method=Image.LANCZOS)

    bg = reference.copy().filter(ImageFilter.GaussianBlur(radius=18))

    x = (ref_w - fitted.size[0]) // 2
    y = (ref_h - fitted.size[1]) // 2
    bg.paste(fitted, (x, y))
    return bg


def _score_similarity(processed: Image.Image, candidate: Image.Image) -> float:
    """
    Much less sensitive score:
    - compare on small downscaled image (512px max side)
    - compare in grayscale (luma)
    - light edge component (also downscaled)
    Lower = more similar.
    """
    # downscale both to same small size
    a = processed.convert("L")
    b = candidate.convert("L")

    a_small = a.copy()
    b_small = b.copy()
    a_small.thumbnail((512, 512), Image.LANCZOS)
    b_small = b_small.resize(a_small.size, Image.LANCZOS)

    a_np = np.array(a_small, dtype=np.int16)
    b_np = np.array(b_small, dtype=np.int16)

    pix = float(np.mean(np.abs(a_np - b_np)))

    # light edge compare (still on small)
    a_e = a_small.filter(ImageFilter.FIND_EDGES)
    b_e = b_small.filter(ImageFilter.FIND_EDGES)
    ae = np.array(a_e, dtype=np.int16)
    be = np.array(b_e, dtype=np.int16)
    edge = float(np.mean(np.abs(ae - be)))

    # weights tuned to avoid false rejections
    return (pix * 1.0) + (edge * 0.25)


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    raw = b""
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # keep detail but avoid huge images
        MAX_DIM = 2560
        if max(original.size) > MAX_DIM:
            original.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

        # deterministic base
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # save deterministic image for AI edit
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # AI edit
        ai_img = _call_ai_edit(tmp_path)

        # force output size/orientation to match uploaded
        ai_img = _match_to_reference_no_crop(ai_img, reference=original)

        # score vs deterministic; reject only if REALLY different
        s = _score_similarity(processed, ai_img)

        # ✅ New sane threshold (your old scoring made 48 look “bad”)
        SCORE_MAX = 50.0

        if s > SCORE_MAX:
            safe = _match_to_reference_no_crop(processed, reference=original)
            return _return_png(
                safe,
                headers={
                    "X-AI-USED": "false",
                    "X-REASON": "diff_gate",
                    "X-SCORE": str(s),
                    "X-SCORE-MAX": str(SCORE_MAX),
                    "X-SIZE": f"{original.size[0]}x{original.size[1]}",
                },
            )

        return _return_png(
            ai_img,
            headers={
                "X-AI-USED": "true",
                "X-SCORE": str(s),
                "X-SCORE-MAX": str(SCORE_MAX),
                "X-SIZE": f"{original.size[0]}x{original.size[1]}",
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
