import io
import os
import base64
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

DO NOT CHANGE (ABSOLUTE):
- Any text, letters, numbers, icons, UI elements, button symbols, screens.
- Wheels/rims/center caps/wheel logos/tire text.
- Badges/logos, grille pattern/shape, headlights/taillights.
- Materials/trim (do NOT add chrome/gloss/metallic; do NOT brighten blacked-out parts).
- Geometry/proportions, background layout, objects.

ALLOWED (GLOBAL ONLY):
- Neutralize color cast / correct white balance
- Mild exposure + contrast improvement
- Mild highlight recovery + mild shadow lift
- Very subtle sharpening + light noise reduction

No local edits. No repainting. Photorealistic. No stylization.
""".strip()


def _to_data_url_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # downscale for stability
        MAX_SIZE = 1536
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # deterministic pre-pass
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        processed_url = _to_data_url_png(processed)

        # Responses API: force EDIT without using gpt-5
        resp = client.responses.create(
            model="chatgpt_image_latest",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": processed_url},
                ],
            }],
            tools=[{
                "type": "image_generation",
                "action": "edit",      # force edit
                "quality": "high",     # more polish
                "size": "auto",        # keep aspect ratio
                "output_format": "png",
            }],
        )

        # pull image result
        img_b64 = None
        for out in getattr(resp, "output", []):
            if getattr(out, "type", None) == "image_generation_call":
                img_b64 = getattr(out, "result", None)
                break

        if not img_b64:
            raise RuntimeError("No image returned from Responses API.")

        out_bytes = base64.b64decode(img_b64)
        return Response(content=out_bytes, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
