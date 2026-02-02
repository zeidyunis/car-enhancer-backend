import base64
import io
import os
import traceback

from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from openai import OpenAI
import openai

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAIN_MODEL = os.getenv("MAIN_MODEL", "gpt-4.1")
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(4000 * 4000)))  # 16MP


PROMPT = """
You MUST edit the provided image. Do NOT generate a new image.

FRAMING (STRICT):
- Keep framing/composition identical (no crop/zoom/rotate).

DO NOT CHANGE (IMMUTABLE):
- Wheels/rims/tires/center caps/center-cap logos
- Badges/logos/emblems
- Grille design
- Headlights/taillights
- Any text/icons/screens/buttons (no warping, keep sharp)

ALLOWED ONLY:
- Neutralize color cast
- Slight exposure/contrast
- Mild highlight recovery
- Subtle sharpness
- Mild noise reduction

Photorealistic and faithful.
""".strip()


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc(), "path": str(request.url)},
    )


def load_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def downscale_if_needed(im: Image.Image) -> Image.Image:
    w, h = im.size
    if (w * h) <= MAX_PIXELS:
        return im
    scale = (MAX_PIXELS / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def pick_tool_size(w: int, h: int) -> str:
    if w > h:
        return "1536x1024"
    if h > w:
        return "1024x1536"
    return "1024x1024"


def to_data_url_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.get("/")
def root():
    return {"ok": True}


@app.get("/version")
def version():
    return {
        "openai_version": getattr(openai, "__version__", "unknown"),
        "has_client_responses": hasattr(client, "responses"),
        "model": MAIN_MODEL,
        "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")

    original = load_image(data)
    safe = downscale_if_needed(original)

    tool_size = pick_tool_size(*safe.size)
    image_url = to_data_url_png(safe)

    if not hasattr(client, "responses"):
        raise HTTPException(500, "OpenAI SDK missing client.responses")

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

    calls = [o for o in resp.output if getattr(o, "type", None) == "image_generation_call"]
    if not calls:
        raise HTTPException(500, "No image_generation_call returned")

    edited = decode_image(calls[0].result)

    # IMPORTANT: DO NOT resize back up (avoids warping artifacts)
    out = io.BytesIO()
    edited.save(out, format="PNG")

    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={
            "x-ai-used": "true",
            "x-tool-size": tool_size,
            "x-returned-size": f"{edited.size[0]}x{edited.size[1]}",
            "cache-control": "no-store",
        },
    )
