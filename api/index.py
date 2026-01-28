import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageFilter
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image


app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"status": "ok"}


def _edge_diff_score(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    a = Image.fromarray(a_rgb).convert("L").filter(ImageFilter.FIND_EDGES)
    b = Image.fromarray(b_rgb).convert("L").filter(ImageFilter.FIND_EDGES)
    a_np = np.array(a, dtype=np.int16)
    b_np = np.array(b, dtype=np.int16)
    return float(np.mean(np.abs(a_np - b_np)))


def _pixel_diff_score(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    a_np = a_rgb.astype(np.int16)
    b_np = b_rgb.astype(np.int16)
    return float(np.mean(np.abs(a_np - b_np)))


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

ALLOWED (GLOBAL ONLY) — MAKE IT NOTICEABLY BETTER BUT REALISTIC
- Neutralize color cast / correct white balance
- Exposure correction (noticeable but not extreme)
- Contrast improvement (noticeable but not HDR)
- Highlight recovery (if possible)
- Mild shadow lift
- Subtle clarity/sharpening (do not create halos)
- Light noise reduction

NO LOCAL EDITS. NO REGION EDITS. NO RETOUCHING SPECIFIC PARTS.
If a change risks altering physical details, do not apply it.

Output must look like the SAME PHOTO, just cleaner and better balanced.
Photorealistic. No stylization.
""".strip()


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # Downscale for stability
        MAX_SIZE = 1536
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # Deterministic baseline
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # Save deterministic image for OpenAI edit
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        result = client.images.edit(
            model="gpt-image-1.5",
            image=open(tmp_path, "rb"),
            prompt=MASTER_PROMPT,
            size="1536x1024",
        )

        out_b64 = result.data[0].b64_json
        out_bytes = base64.b64decode(out_b64)

        ai_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")

        # Resize AI output to match processed size for scoring + final output
        if ai_img.size != processed.size:
            ai_img = ai_img.resize(processed.size, Image.LANCZOS)

        ai_np = np.array(ai_img, dtype=np.uint8)

        edge_score = _edge_diff_score(processed_np, ai_np)
        pix_score = _pixel_diff_score(processed_np, ai_np)

        # ✅ Looser so AI is allowed more often
        EDGE_MAX = 8.0
        PIX_MAX = 12.0

        if edge_score > EDGE_MAX or pix_score > PIX_MAX:
            buf = io.BytesIO()
            processed.save(buf, format="PNG")
            return Response(
                content=buf.getvalue(),
                media_type="image/png",
                headers={
                    "X-AI-USED": "false",
                    "X-EDGE-DIFF": str(edge_score),
                    "X-PIX-DIFF": str(pix_score),
                },
            )

        buf = io.BytesIO()
        ai_img.save(buf, format="PNG")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-AI-USED": "true",
                "X-EDGE-DIFF": str(edge_score),
                "X-PIX-DIFF": str(pix_score),
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
