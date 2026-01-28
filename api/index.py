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

=====================
ABSOLUTE IMMUTABLE RULES (MUST FOLLOW)
=====================

IDENTITY & GEOMETRY
- Preserve the exact car identity: model, trim, generation, body shape.
- Preserve exact proportions, panel gaps, body lines, window shapes.
- Do NOT warp, stretch, reshape, or redraw any part.

WHEELS & LOGOS
- Do NOT modify wheels in any way.
- Do NOT change rim design, finish, color, or texture.
- Do NOT change tire sidewall text.
- Do NOT blur, redraw, replace, sharpen, or reinterpret center caps.
- Wheel logos and brand marks must remain EXACTLY the same pixels.

BADGES & BRANDING
- Do NOT change, redraw, blur, sharpen, replace, or reinterpret any badges.
- Do NOT modify brand logos anywhere on the car or interior.

LIGHTS
- Do NOT change headlights, taillights, indicators, DRLs, or reflectors.
- Do NOT alter lens texture, LED patterns, or housing shape.

MATERIALS & TRIM
- Do NOT add chrome, gloss, metallic, or reflective trim.
- Do NOT convert blacked-out parts into chrome or bright materials.
- Do NOT change matte ↔ gloss finishes.
- Do NOT change carbon fiber, piano black, wood, or aluminum textures.
- Do NOT recolor interior or exterior materials.

INTERIOR UI / TEXT / ICONS
- Do NOT change any text, letters, numbers, symbols, icons, or fonts.
- Do NOT alter dashboard controls, buttons, climate controls, steering buttons.
- Do NOT change infotainment screens, instrument cluster, HUD, or displays.
- Do NOT blur, redraw, or stylize any UI elements.

BACKGROUND & ENVIRONMENT
- Do NOT add, remove, or modify objects.
- Do NOT change background layout, walls, reflections, scenery, or shadows.
- Do NOT add people, cars, signs, or props.

REFLECTIONS
- Preserve original reflections structure.
- Do NOT invent new reflections or highlights.

=====================
ALLOWED OPERATIONS (GLOBAL ONLY)
=====================

You may apply ONLY subtle, global, uniform adjustments:

- Neutralize color cast / correct white balance
- Slight exposure correction
- Slight contrast improvement
- Very mild highlight recovery
- Very mild shadow lift
- Very subtle sharpening
- Very light noise reduction

These adjustments must affect the ENTIRE image evenly.
No local edits. No region-specific edits.

=====================
STRICT PROHIBITIONS
=====================

- No repainting
- No retouching of specific parts
- No enhancement of individual components
- No stylization
- No HDR look
- No cinematic grading
- No “AI look”
- No reinterpretation
- No reconstruction
- No artistic effects

If any enhancement risks changing physical details,
you must choose to leave the image unchanged.

=====================
OUTPUT REQUIREMENT
=====================

The output must look like the SAME PHOTO,
taken with better lighting and camera settings,
not a different version of the car.

Photorealistic.
Neutral.
Conservative.
Technically accurate.
No creative interpretation.

Failure to follow any rule is incorrect.
""".strip()


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # reduce size (helps stability + reduces micro-detail repainting)
        MAX_SIZE = 1536
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # deterministic step
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # save for OpenAI edit
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # AI polish
        result = client.images.edit(
            model="gpt-image-1.5",
            image=open(tmp_path, "rb"),
            prompt=MASTER_PROMPT,
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
