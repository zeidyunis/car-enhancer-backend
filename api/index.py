import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image


app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {"status": "ok", "openai_key_present": has_key}


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()

        image = Image.open(io.BytesIO(raw)).convert("RGB")

        processed = enhance_image(image)  # numpy RGB
        pil_img = Image.fromarray(processed.astype(np.uint8), mode="RGB")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        pil_img.save(tmp_path)

        prompt = """
Enhance this photo for a car sales listing.

Rules:
- Do NOT change car model, shape, wheels, badges
- Do NOT add/remove objects
- Do NOT alter background
- Keep reflections natural

Only:
- Improve lighting
- Improve contrast
- Improve clarity
- Reduce color cast
- Subtle sharpening

Photorealistic.
"""

        result = client.images.edit(
            model="chatgpt-image-latest",
            image=open(tmp_path, "rb"),
            prompt=prompt,
            size="1024x1024"
        )

        img_base64 = result.data[0].b64_json
        final_bytes = base64.b64decode(img_base64)

        return JSONResponse({"image_base64": base64.b64encode(final_bytes).decode()})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )
