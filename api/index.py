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

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


@app.get("/")
def root():
    return {"status": "ok"}


MASTER_PROMPT = """
You are performing a STRICT photo enhancement, not a redesign.

Your task is to apply ONLY global photographic corrections.
You are NOT allowed to change any physical, visual, or material details.

ABSOLUTE IMMUTABLE RULES
- Preserve exact car identity and geometry.
- Do NOT modify wheels, rims, center caps, or tire text.
- Do NOT modify badges or logos.
- Do NOT modify headlights or taillights.
- Do NOT add chrome, gloss, or metallic trim.
- Do NOT change blacked-out parts.
- Do NOT change interior text, icons, or screens.
- Do NOT add/remove objects.
- Preserve original reflections.

ALLOWED (GLOBAL ONLY)
- Neutralize white balance
- Improve exposure
- Improve contrast
- Mild highlight recovery
- Mild shadow lift
- Subtle sharpening
- Light noise reduction

NO LOCAL EDITS.
NO REDRAWING.
NO REPAINTING.

Output must look like the SAME photo, only cleaner and better balanced.
Photorealistic. No stylization.
""".strip()


def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()

        # Load image
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # Downscale for stability (prevents AI from repainting micro-details)
        MAX_SIZE = 1536
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # Deterministic baseline (no hallucination)
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # Convert to data URL for Responses API
        processed_data_url = pil_to_data_url(processed)

        # Call OpenAI in forced EDIT mode, high quality
        resp = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": MASTER_PROMPT
                        },
                        {
                            "type": "input_image",
                            "image_url": processed_data_url
                        },
                    ],
                }
            ],
            tools=[
                {
                    "type": "image_generation",
                    "action": "edit",
                    "quality": "high",
                    "size": "auto",
                    "format": "png",
                }
            ],
        )

        # Extract image result
        image_b64 = None

        for out in getattr(resp, "output", []):
            if getattr(out, "type", None) == "image_generation_call":
                image_b64 = getattr(out, "result", None)
                break

        if not image_b64:
            raise RuntimeError("No image returned from OpenAI.")

        out_bytes = base64.b64decode(image_b64)

        return Response(
            content=out_bytes,
            media_type="image/png"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        )
