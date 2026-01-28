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


MASTER_PROMPT = """
You are performing a STRICT photo enhancement, not a redesign.

Your task is to apply ONLY global photographic corrections.
You are NOT allowed to change any physical, visual, or material details.

ABSOLUTE IMMUTABLE RULES (MUST FOLLOW)
- Preserve exact car identity and geometry (no warping, reshaping, redrawing).
- Do NOT modify wheels in any way (rim design, tire text, center caps, wheel logos).
- Do NOT modify badges/logos anywhere.
- Do NOT modify headlights/taillights/LED patterns/housings.
- Do NOT add chrome/gloss/metallic trim; do NOT turn blacked-out parts into chrome.
- Do NOT change any interior text/icons/UI/screens/buttons.
- Do NOT add/remove objects; do NOT change background layout.
- Preserve reflections structure; do NOT invent reflections.

ALLOWED (GLOBAL ONLY)
- Neutralize color cast / correct white balance
- Slight exposure correction
- Slight contrast improvement
- Very mild highlight recovery
- Very mild shadow lift
- Very subtle sharpening
- Very light noise reduction

NO LOCAL EDITS. NO REGION EDITS.
If a change risks altering physical details, do not apply it.

Output must look like the SAME PHOTO, just cleaner and better balanced.
Photorealistic. No stylization.
""".strip()


def _to_data_url_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # Downscale for stability
        MAX_SIZE = 1536
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # Deterministic step first
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # Put the *processed image* into context and FORCE an EDIT
        processed_data_url = _to_data_url_png(processed)

        resp = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": MASTER_PROMPT},
                        {"type": "input_image", "image_url": processed_data_url},
                    ],
                }
            ],
            tools=[{"type": "image_generation", "action": "edit"}],
        )

        # Extract image result from response
        img_b64 = None
        for out in getattr(resp, "output", []):
            if getattr(out, "type", None) == "image_generation_call":
                img_b64 = getattr(out, "result", None)
                break

        if not img_b64:
            raise RuntimeError("No image_generation_call result returned from Responses API.")

        out_bytes = base64.b64decode(img_b64)

        return Response(content=out_bytes, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
