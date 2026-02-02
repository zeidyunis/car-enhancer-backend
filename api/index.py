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
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(4000 * 4000)))  # 16MP cap


PROMPT = """
You MUST edit the provided image. Do NOT generate a new image.

Keep framing identical (no crop/zoom/rotate).
Do not change wheels, wheel logos, badges, grille, headlights, text/icons.
Only do pro global corrections: WB/color cast, exposure, contrast, highlights/shadows, mild clarity/sharpness.
Photorealistic. Faithful.
""".strip()


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc()},
    )


def _load_rgb(data: bytes) -> Image.Image:
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


def _pick_size(w: int, h: int) -> str:
    # tool presets
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


def _decode_b64_png(b64: str) -> bytes:
    return base64.b64decode(b64)


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
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    img = _downscale_if_needed(_load_rgb(data))
    tool_size = _pick_size(*img.size)
    image_url = _to_data_url_png(img)

    if not hasattr(client, "responses"):
        raise HTTPException(status_code=500, detail="client.responses missing (wrong SDK deployed)")

    resp = client.responses.create(
        model=MAIN_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT},
                {"type": "input_image", "image_url": image_url},
            ],
        }],
        tools=[{
            "type": "image_generation",
            "action": "edit",              # FORCE EDIT
            "input_fidelity": "high",
            "size": tool_size,
            "quality": "high",
        }],
    )

    calls = [o for o in resp.output if getattr(o, "type", None) == "image_generation_call"]
    if not calls:
        raise HTTPException(status_code=500, detail="No image_generation_call returned")

    call0 = calls[0]
    png_bytes = _decode_b64_png(call0.result)

    # IMPORTANT: return tool output as-is (no upscaling back -> fewer warps)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "x-ai-used": "true",
            "x-tool-size": tool_size,
            "x-action": str(getattr(call0, "action", "unknown")),
            "cache-control": "no-store",
        },
    )
