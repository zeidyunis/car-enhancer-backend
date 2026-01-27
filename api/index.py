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

        original_pil = Image.open(io.BytesIO(raw)).convert("RGB")
        original_np = np.array(original_pil)

        # deterministic step
        enhanced_np, orig_np, lock_fn = enhance_image(original_pil)
        enhanced_pil = Image.fromarray(enhanced_np.astype(np.uint8))

        # save temp for AI
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        enhanced_pil.save(tmp_path)

        prompt = """
Enhance this photo for a car listing.
Preserve all geometry, logos, wheels, headlights.
No repainting. No replacement.
Only color/light balance.
"""

        # AI polish
        result = client.images.edit(
            model="gpt-image-1",
            image=open(tmp_path, "rb"),
            prompt=prompt,
            size="1024x1024"
        )

        out_b64 = result.data[0].b64_json
        ai_img = base64.b64decode(out_b64)

        ai_np = np.array(Image.open(io.BytesIO(ai_img)).convert("RGB"))

        # HARD LOCK: restore protected regions
        final = lock_fn(orig_np, ai_np)

        final_pil = Image.fromarray(final.astype(np.uint8))

        buf = io.BytesIO()
        final_pil.save(buf, format="PNG")

        return Response(buf.getvalue(), media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
