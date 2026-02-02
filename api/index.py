import base64
import io
import os
import traceback

from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from PIL import Image, ImageOps
from openai import OpenAI
import openai


# --------------------
# App + Client
# --------------------

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MAIN_MODEL = os.getenv("MAIN_MODEL", "gpt-4.1")

# Max allowed pixels (16MP default)
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(4000 * 4000)))


# --------------------
# Prompt (Hard Lock)
# --------------------

PROMPT = """
Edit (NOT generate) this exact photo for a car listing.

STRICT RULES:
- This is an EDIT, not a new image.
- Do NOT change wheels, rims, center caps, logos, badges.
- Do NOT change grille design.
- Do NOT change headlights/taillights.
- Do NOT warp text, icons, screens.
- Do NOT add chrome or new trim.
- Do NOT change body shape.
- Do NOT repaint objects.

ALLOWED ONLY:
- Fix color cast
- Slight exposure/contrast
- Mild highlight recovery
- Subtle sharpness
- Light noise reduction

Keep everything photorealistic and identical.
"""


# --------------------
# Helpers
# --------------------

def load_image(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")


def downscale_if_needed(im: Image.Image) -> Image.Image:
    w, h = im.size

    if (w * h) <= MAX_PIXELS:
        return im

    scale = (MAX_PIXELS / float(w * h)) ** 0.5
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def pick_tool_size(w: int, h: int) -> str:
    if w > h:
        return "1536x1024"
    if h > w:
        return "1024x1536"
    return "1024x1024"


def pil_to_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# --------------------
# Debug Root
# --------------------

@app.get("/")
def root():
    return {
        "ok": True,
        "openai_version": getattr(openai, "__version__", "unknown"),
        "has_client_responses": hasattr(client, "responses"),
        "model": MAIN_MODEL,
    }


# --------------------
# Main Endpoint
# --------------------

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):

    try:

        # --------------------
        # Read upload
        # --------------------

        data = await file.read()

        if not data:
            raise HTTPException(400, "Empty upload")


        # --------------------
        # Load + preprocess
        # --------------------

        original = load_image(data)
        orig_w, orig_h = original.size

        safe = downscale_if_needed(original)

        tool_size = pick_tool_size(*safe.size)

        image_url = pil_to_data_url(safe)


        # --------------------
        # Guard SDK version
        # --------------------

        if not hasattr(client, "responses"):
            raise HTTPException(
                500,
                "OpenAI SDK too old. client.responses missing."
            )


        # --------------------
        # FORCE EDIT (Responses API)
        # --------------------

        resp = client.responses.create(
            model=MAIN_MODEL,

            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PROMPT},
                        {"type": "input_image", "image_url": image_url},
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


        # --------------------
        # Extract result
        # --------------------

        calls = [
            o for o in resp.output
            if getattr(o, "type", None) == "image_generation_call"
        ]

        if not calls:
            raise HTTPException(500, "No image_generation_call returned")

        call0 = calls[0]

        edited = decode_image(call0.result)


        # --------------------
        # Resize back to original
        # --------------------

        edited = edited.resize(
            (orig_w, orig_h),
            Image.Resampling.LANCZOS
        )


        # --------------------
        # Return PNG
        # --------------------

        out = io.BytesIO()
        edited.save(out, format="PNG")

        return Response(
            content=out.getvalue(),
            media_type="image/png",
            headers={
                "x-ai-used": "true",
                "x-action": str(getattr(call0, "action", "unknown")),
                "x-orig-size": f"{orig_w}x{orig_h}",
                "x-tool-size": tool_size,
                "cache-control": "no-store",
            },
        )


    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            500,
            f"{str(e)}\n{traceback.format_exc()}"
        )
