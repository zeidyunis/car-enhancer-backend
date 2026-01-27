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
        MAX_SIZE = 2048
if max(original.size) > MAX_SIZE:
    original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        processed_np = enhance_image(original)
        processed = Image.fromarray(processed_np.astype(np.uint8), mode="RGB")

        # save BOTH images
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

ABSOLUTE RULES (MUST FOLLOW):
- Preserve the exact car identity: same model/trim/badges
- Preserve exact geometry: body lines, proportions, windows, panel gaps
- DO NOT modify wheels AT ALL:
  - Do not change rim design
  - Do not change tire sidewall text
  - Do not change center caps
  - Do not change wheel logos/brand marks
  - Wheel logos must remain EXACTLY the same (no redraw, no blur, no replacement)
- Do NOT add/remove objects, text, logos, plates, people
- Do NOT change background layout
- If any rule conflicts with enhancement, prioritize NOT changing anything.

Allowed edits ONLY (global, subtle):
- Neutralize color cast
- Slightly deepen blacks
- Slight highlight recovery
- Mild contrast
- Very subtle clarity/sharpening (no HDR)

Output must look like the SAME photo, only cleaner.
Photorealistic. No stylization.
"""

        result = client.images.edit(
            model="gpt-image-1.5",
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
