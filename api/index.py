import base64
import io
import os
from typing import Tuple

from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from PIL import Image, ImageOps
from openai import OpenAI

app = FastAPI()
client = OpenAI()

# Main “controller” model for Responses API (NOT the image model).
# Docs show gpt-4.1 works with image_generation tool. :contentReference[oaicite:3]{index=3}
MAIN_MODEL = os.getenv("MAIN_MODEL", "gpt-5")

# Hard cap to protect your Vercel function time/memory.
# (Vercel also has request body limits; huge uploads may 413 before reaching code.)
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(4000 * 4000)))  # 16MP default


PROMPT = """Enhance this exact photo for an online car sales listing.

CRITICAL RULES (MUST FOLLOW):
- This is an EDIT of the provided photo, NOT a new generated image.
- Do NOT change the car identity: model, trim, headlights, grille design, wheels, badges, logos, text, reflections, tint level.
- Do NOT add/remove objects, do NOT invent chrome trims, do NOT modify body lines, panel gaps, vents, emblems.
- Preserve ALL readable text and symbols (wheel center caps, dashboard buttons, brand marks, license plate text if present).
- Keep geometry stable: no warping, no stretching, no reshaping.

Allowed adjustments ONLY:
- Correct minor lens distortion and perspective subtly (keep proportions realistic).
- Neutralize color cast (especially fluorescent/green/yellow indoor cast).
- Slightly deepen blacks, recover highlights, reduce blown lights.
- Subtle clarity/sharpness improvement without halos.
- Gentle contrast and vibrance improvements, realistic.

Output must remain photorealistic and faithful to the input image.
"""


def _load_image(file_bytes: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(file_bytes))
        # Respect EXIF orientation so we don't “cut” / rotate wrong
        im = ImageOps.exif_transpose(im)
        return im
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Upload a valid PNG/JPEG/WEBP.")


def _maybe_downscale(im: Image.Image) -> Image.Image:
    w, h = im.size
    if (w * h) <= MAX_PIXELS:
        return im
    # scale down uniformly
    scale = (MAX_PIXELS / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _pick_tool_size(w: int, h: int) -> str:
    # Available tool sizes: 1024x1024, 1536x1024, 1024x1536, or auto :contentReference[oaicite:4]{index=4}
    if w == h:
        return "1024x1024"
    return "1536x1024" if w > h else "1024x1536"


def _pil_to_data_url_png(im: Image.Image) -> str:
    # Use PNG for maximal fidelity into the model
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _decode_tool_image_base64(image_b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=500, detail="OpenAI returned invalid image data.")


@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/enhance (POST multipart/form-data file=...)"]}


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    # 1) Read upload
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    # 2) Load + orient + optional downscale
    original = _load_image(data).convert("RGB")
    orig_w, orig_h = original.size

    safe = _maybe_downscale(original)
    tool_size = _pick_tool_size(*safe.size)

    # 3) Prepare image for Responses API (image must be in context when forcing edit) :contentReference[oaicite:5]{index=5}
    img_url = _pil_to_data_url_png(safe)

    # 4) FORCE EDIT via Responses API image_generation tool with action:"edit" :contentReference[oaicite:6]{index=6}
    try:
        resp = client.responses.create(
            model=MAIN_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PROMPT},
                        {"type": "input_image", "image_url": img_url},
                    ],
                }
            ],
            tools=[
                {
                    "type": "image_generation",
                    "action": "edit",
                    "input_fidelity": "high",
                    "size": tool_size,
                    "quality": "high",
                }
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

    # 5) Extract tool output
    calls = [o for o in resp.output if getattr(o, "type", None) == "image_generation_call"]
    if not calls:
        raise HTTPException(status_code=500, detail="No image_generation_call in response.")
    call0 = calls[0]
    image_b64 = call0.result
    action_used = getattr(call0, "action", "unknown")

    # 6) Decode and RESIZE BACK to original exact dimensions (prevents “cutting”)
    edited = _decode_tool_image_base64(image_b64)

    # If we downscaled before tool, upscale back to the original size
    edited = edited.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    # 7) Return PNG bytes
    out_buf = io.BytesIO()
    edited.save(out_buf, format="PNG")
    out_bytes = out_buf.getvalue()

    headers = {
        "x-ai-used": "true",
        "x-action": str(action_used),
        "x-orig-size": f"{orig_w}x{orig_h}",
        "x-tool-size": tool_size,
        "cache-control": "no-store",
    }

    return Response(content=out_bytes, media_type="image/png", headers=headers)
