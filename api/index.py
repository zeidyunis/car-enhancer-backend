import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image


app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"status": "ok"}


PROMPT = """
Enhance this photo for a car sales listing.

STRICT RULES:
- Keep the car, rims, badges, logos, headlights, taillights, interior buttons, screens, and text EXACTLY the same.
- Do NOT add chrome, gloss, metallic trim, or reflections.
- Do NOT recolor blacked-out or matte parts.
- Do NOT change materials, geometry, or proportions.
- Do NOT add/remove objects or background elements.

ONLY DO:
- Correct white balance / remove color cast
- Improve exposure and contrast naturally
- Recover highlights if needed
- Subtle sharpening (no halos)
- Light noise reduction

Photorealistic. No stylization.
No redesign. No repainting.
""".strip()


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # Reduce size (helps stability + limits hallucinations)
        MAX_SIZE = 1536
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # Deterministic pre-processing
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # Save temp file for OpenAI
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # AI polish (stable conservative model)
        result = client.images.edit(
            model="gpt-image-1.5",
            image=open(tmp_path, "rb"),
            prompt=PROMPT,
            size="auto"
        )

        out_b64 = result.data[0].b64_json
        out_bytes = base64.b64decode(out_b64)

        return Response(content=out_bytes, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
