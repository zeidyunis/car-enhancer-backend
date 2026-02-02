import io
import os
import base64
import tempfile
import traceback

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageEnhance
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"status": "ok"}


PROMPT = """
Perform only a photo edit (not a recomposition or artistic reinterpretation)
to improve this car photo for a car sales listing.

ABSOLUTE RULES:
Do NOT change:
- Wheels/rims/spokes/center caps/tire text
- Badges, brand logos, or emblems anywhere
- Grille pattern, shape, or texture
- Headlights, DRLs, taillights, or internal lighting structure
- Any text, icons, UI buttons, or screens
- Materials/trim (no added chrome/gloss, no brightening blacked-out surfaces)
- Body shape, reflections, shadows, background layout

ALLOWED GLOBAL CORRECTIONS:
- White balance / remove color cast
- Exposure + contrast correction (natural)
- Mild highlight recovery
- Mild shadow lift
- Very light noise reduction

NO local retouching, no “beautify”, no HDR, no stylization.
Photorealistic.
""".strip()


def _return_png(img: Image.Image, headers: dict) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # keep detail but avoid huge images
        MAX_DIM = 2560
        if max(original.size) > MAX_DIM:
            original.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

        # deterministic cleanup
        processed = Image.fromarray(enhance_image(original).astype("uint8"))

        # save temp for AI
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # EDIT via OpenAI (forces an *edit*, not a pure regeneration)
        result = client.images.edit(
            model="gpt-image-1",
            image=open(tmp_path, "rb"),
            prompt=PROMPT,
            size="auto",     # keep original aspect ratio
        )

        out_b64 = result.data[0].b64_json
        out_bytes = base64.b64decode(out_b64)

        # mild “anti-matte” polish (keeps it realistic)
        ai_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")
        ai_img = ImageEnhance.Contrast(ai_img).enhance(1.04)
        ai_img = ImageEnhance.Color(ai_img).enhance(1.02)

        return _return_png(
            ai_img,
            headers={
                "X-AI-USED": "true",
                "X-OUT-SIZE": f"{ai_img.size[0]}x{ai_img.size[1]}",
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
