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

# Create client even if key is missing; we’ll validate on use.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAIN_MODEL = os.getenv("MAIN_MODEL", "gpt-4.1")
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(4000 * 4000)))  # 16MP


PROMPT = """
You MUST edit the provided image. Do NOT generate a new image.

FRAMING (STRICT):
- Keep framing/composition identical (no crop/zoom/rotate).
- Keep aspect ratio identical.

DO NOT CHANGE (IMMUTABLE):
- Wheels/rims/spokes/tires/center caps/center-cap logos
- Badges/logos/emblems anywhere
- Grille pattern/mesh/shape/texture
- Headlights/taillights/DRL shapes and inner structure
- Any text/numbers/icons/screens/buttons (must remain sharp and unwarped)
- Body shape, panel lines, reflections geometry, tint level
- Background objects/layout
- Trim/materials: do NOT add chrome, do NOT change blacked-out trim, do NOT change matte↔gloss

ALLOWED (GLOBAL ONLY):
- Neutralize color cast / white balance
- Slight exposure + contrast improvement (natural)
- Mild highlight recovery, mild shadow lift
- Very subtle sharpness/clarity (no halos)
- Mild noise reduction only if needed

Photorealistic. Faithful to input.
""".strip()


@app.exception_handler(Exception)
async def _catch_all(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "trace": traceback.format_exc(),
            "path": str(request.url),
        },
    )


def _load_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def _downscale_if_needed(im: Image.Image) -> Image.Image:
    w, h = im.size
    if (w * h) <= MAX_PIXELS:
        return im
    scale = (MAX_PIXELS / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _pick_tool_size(w: int, h: int) -> str:
    if w > h:
        return "1536x1024"
    if h > w:
        return "1024x1536"
    return "1024x1024"


def _to_data_url_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _decode_b64_image(b64: str) -> Image.Image:
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
        "main_model": MAIN_MODEL,
        "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    # hard fail with clear message if key missing
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY env var on Vercel.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    original = _load_image(data)
    orig_w, orig_h = original.size

    safe = _downscale_if_needed(original)
    tool_size = _pick_tool_size(*safe.size)
    image_url = _to_data_url_png(safe)

    if not hasattr(client, "responses"):
        raise HTTPException(
            status_code=500,
            detail="OpenAI SDK too old on this deployment (no client.responses). Fix requirements.txt + redeploy with Clear Cache.",
        )

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
        raise HTTPException(status_code=500, detail="No image_generation_call returned.")

    edited = _decode_b64_image(calls[0].result)
    edited = edited.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    out = io.BytesIO()
    edited.save(out, format="PNG")

    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={
            "x-ai-used": "true",
            "x-orig-size": f"{orig_w}x{orig_h}",
            "x-tool-size": tool_size,
            "cache-control": "no-store",
        },
    )
