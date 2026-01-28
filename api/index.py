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
Enhance this photo for a car sales listing.

ABSOLUTE RULES:
- Do NOT change wheels, rims, spokes, center caps, or wheel logos.
- Do NOT change grille pattern, shape, or texture.
- Do NOT change badges or brand logos.
- Do NOT change any text, numbers, icons, screens, or buttons.
- Do NOT add chrome, gloss, metallic trim, or new reflections.
- Do NOT recolor blacked-out or matte parts.
- Do NOT change materials, geometry, or proportions.
- Do NOT add/remove objects or background elements.

ONLY DO (GLOBAL ONLY):
- Correct white balance / remove color cast
- Improve exposure and contrast naturally
- Recover highlights if needed
- Slightly deepen blacks (do not crush)
- Add mild clarity (avoid halos)
- Light noise reduction

Photorealistic. No stylization. No repainting.
""".strip()


def post_polish(pil_img: Image.Image) -> Image.Image:
    """
    Fix the 'matte/blurry' vibe safely (global only).
    Very conservative: adds a touch of contrast + clarity without changing details.
    """
    img = pil_img.convert("RGB")

    # Slight contrast (reduces matte/flat look)
    img = ImageEnhance.Contrast(img).enhance(1.06)

    # Tiny black point / depth (very mild)
    img = ImageEnhance.Brightness(img).enhance(0.99)

    # Mild clarity using unsharp mask (safe global sharpening)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=3))

    return img


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # limit size for stability (but keep aspect ratio)
        MAX_SIZE = 2048
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # deterministic cleanup (safe)
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # save temp
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # AI edit (keep aspect ratio)
        result = client.images.edit(
            model="gpt-image-1",
            image=open(tmp_path, "rb"),
            prompt=PROMPT,
            size="auto",  # ✅ important: avoids square resize blur
        )

        out_b64 = result.data[0].b64_json
        out_bytes = base64.b64decode(out_b64)

        # post-polish to remove matte/blurry feel
        ai_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")
        final_img = post_polish(ai_img)

        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
