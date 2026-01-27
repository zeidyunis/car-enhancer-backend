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

        # deterministic step
        processed_np = enhance_image(original)
        processed = Image.fromarray(processed_np.astype(np.uint8), mode="RGB")

        # save BOTH images for edit conditioning
        orig_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        orig_path = orig_tmp.name
        orig_tmp.close()
        original.save(orig_path)

        proc_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        proc_path = proc_tmp.name
        proc_tmp.close()
        processed.save(proc_path)

        prompt = """
Edit the FIRST image using the SECOND image only as a reference for color/lighting.

ABSOLUTE RULES (must follow):
- Preserve the exact car identity: same model, trim, badges, wheels, grille, headlights
- Preserve exact geometry: body lines, proportions, wheel shape, window shape, panel gaps
- Do NOT add/remove objects, text, logos, plates, people
- Do NOT change background layout or reflections structure
- No repainting, no new rims, no tint change

Allowed edits ONLY:
- Neutralize color cast (make whites neutral)
- Slightly deepen blacks
- Recover highlights if possible
- Mild contrast improvement
- Very subtle clarity/sharpening (no HDR look)

Output must look like the SAME photo, just cleaner and more balanced.
Photorealistic. No stylization.
"""

        # IMPORTANT: pass both images (primary + reference)
        result = client.images.edit(
            model="chatgpt-image-latest",
            image=[open(orig_path, "rb"), open(proc_path, "rb")],
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
