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


# Loosened just enough to allow real edits, still protects your problem areas.
PROMPT = """
Enhance this exact photo for a car sales listing. Improve the photo quality while preserving identity.

DO NOT CHANGE (ABSOLUTE):
- Wheels/rims/spokes/tires/center caps/center-cap logos
- Badges/logos anywhere
- Grille pattern/mesh/shape/texture
- Headlights/taillights/DRL shapes and internal patterns
- Any text/numbers/icons/screens/buttons (especially interior controls)
- Body shape, reflections structure, background layout/objects
- Materials/trim (do NOT add chrome, do NOT brighten blacked-out trim)

ALLOWED (GLOBAL PHOTO CORRECTIONS ONLY):
- Better white balance / remove color cast
- Improve exposure and contrast (natural, not HDR)
- Slightly deepen blacks and improve midtone contrast
- Mild highlight recovery
- Mild shadow lift
- Mild noise reduction

Photorealistic. No stylization. No local retouching.
""".strip()


def _safe_polish(img: Image.Image) -> Image.Image:
    # Mild anti-matte polish (clean, no halos)
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.07)
    img = ImageEnhance.Color(img).enhance(1.03)
    img = ImageEnhance.Sharpness(img).enhance(1.05)
    return img


def _return_png(img: Image.Image, headers: dict) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    raw = b""
    tmp_path = None
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # deterministic base
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # temp file for OpenAI
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # AI edit (ALWAYS return AI result in this build)
        result = client.images.edit(
            model="gpt-image-1",
            image=open(tmp_path, "rb"),
            prompt=PROMPT,
            size="auto",
        )

        out_bytes = base64.b64decode(result.data[0].b64_json)
        ai_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")

        final = _safe_polish(ai_img)

        return _return_png(
            final,
            headers={
                "X-AI-USED": "true",
                "X-IN-WH": f"{original.size[0]}x{original.size[1]}",
                "X-PROCESSED-WH": f"{processed.size[0]}x{processed.size[1]}",
                "X-AI-WH": f"{ai_img.size[0]}x{ai_img.size[1]}",
            },
        )

    except Exception as e:
        msg = str(e)

        # If billing is blocking, you were seeing "no edits" because AI never ran.
        if "billing_hard_limit" in msg or "hard limit" in msg or "billing" in msg:
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                processed_np = enhance_image(img).astype(np.uint8)
                processed = Image.fromarray(processed_np, mode="RGB")
                fallback = _safe_polish(processed)
                return _return_png(
                    fallback,
                    headers={"X-AI-USED": "false", "X-REASON": "billing"},
                )
            except Exception:
                pass

        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
