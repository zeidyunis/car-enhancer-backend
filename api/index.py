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


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # downscale (helps size + speed)
        MAX_SIZE = 2048
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # deterministic step (pillow-only)
        processed_np = enhance_image(original)  # returns numpy RGB array
        processed = Image.fromarray(processed_np.astype(np.uint8), mode="RGB")

        # save for OpenAI edit
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        prompt = """
Make ONLY global photo corrections. Do NOT make any local edits.

ABSOLUTE LOCKS:
- Do not change any text, letters, numbers, icons, logos, UI elements, button symbols, stitching patterns, or screen contents.
- Do not warp or redraw dashboard controls, climate controls, infotainment UI, or instrument cluster.
- Do not alter interior material colors (leather/trim) beyond neutral white balance.

Allowed ONLY (global):
- neutral white balance / remove color cast
- very mild exposure adjustment
- very mild contrast
- very mild highlight recovery
- very mild noise reduction
NO local enhancements. NO repainting. Photorealistic.
"""
        result = client.images.edit(
            model="gpt-image-1.5",
            image=open(tmp_path, "rb"),
            prompt=prompt,
            size="1536x1024"
        )

        out_b64 = result.data[0].b64_json
        out_bytes = base64.b64decode(out_b64)

        return Response(content=out_bytes, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
