import io
import os
import base64
import tempfile

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from openai import OpenAI

from utils.opencv_pipeline import enhance_image


app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):

    # ---------- Read Upload ----------
    raw = await file.read()

    image = Image.open(io.BytesIO(raw)).convert("RGB")

    # ---------- OpenCV Pipeline ----------
    processed = enhance_image(image)

    pil_img = Image.fromarray(processed)

    # ---------- Save Temp ----------
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_img.save(tmp.name)

    # ---------- OpenAI Polish ----------
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
        model="gpt-image-latest",
        image=open(tmp.name, "rb"),
        prompt=prompt,
        size="1024x1024"
    )

    img_base64 = result.data[0].b64_json

    final_bytes = base64.b64decode(img_base64)

    return JSONResponse({
        "image_base64": base64.b64encode(final_bytes).decode()
    })
