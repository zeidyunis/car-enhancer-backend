import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageEnhance
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
- Any text, letters, numbers, icons, UI elements, screens, or buttons (especially interior controls).
- Body shape, reflections structure, background objects/layout.

ALLOWED (GLOBAL PHOTO CORRECTIONS ONLY):
- Correct white balance / remove fluorescent color cast
- Improve exposure + contrast (natural, not HDR)
- Recover highlights a bit (no fake reflections)
- Lift shadows slightly (no “washed” look)
- Mild noise reduction
- Very subtle crispness (no halos)

IMPORTANT:
- No local edits. No retouching individual parts.
- If any enhancement risks changing physical details, DO NOT APPLY it.
- Prefer returning a minimally changed image over altering any details.

Photorealistic. No stylization.
""".strip()


def post_polish_safe(img: Image.Image) -> Image.Image:
    """
    Slightly stronger global polish to reduce 'matte' but still clean.
    No unsharp mask (avoids halos/sloppy look).
    """
    img = img.convert("RGB")

    # Reduce matte: small contrast + tiny saturation
    img = ImageEnhance.Contrast(img).enhance(1.06)
    img = ImageEnhance.Color(img).enhance(1.03)       # tiny vibrance (safe)

    # Very mild crispness (Pillow sharpness is safer than unsharp mask)
    img = ImageEnhance.Sharpness(img).enhance(1.06)

    return img


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # Keep more detail (helps reduce "matte"), but still control size for Vercel stability
        MAX_SIZE = 2560
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # Deterministic pre-processing (your pipeline)
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # Save temp for OpenAI edit
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # AI edit (keep aspect ratio to avoid resize blur)
        result = client.images.edit(
            model="gpt-image-1.5",
            image=open(tmp_path, "rb"),
            prompt=PROMPT,
            size="auto",
        )

        out_bytes = base64.b64decode(result.data[0].b64_json)

        # Global polish to reduce matte look
        ai_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")
        final_img = post_polish_safe(ai_img)

        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
