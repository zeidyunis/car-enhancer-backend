import base64
import io
import os
import traceback

from fastapi import FastAPI, File, UploadFile, Response, HTTPException
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


def _load_image(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)  # iPhone orientation fix
        return im.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image upload.")


def _downscale_if_needed(im: Image.Image) -> Image.Image:
    w, h = im.size
    if (w * h) <= MAX_PIXELS:
        return im
    scale = (MAX_PIXELS / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _pick_tool_size(w: int, h: int) -> str:
    # OpenAI image tool sizes (safe presets)
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
    }


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload.")

        original = _load_image(data)
        orig_w, orig_h = original.size

        safe = _downscale_if_needed(original)
        tool_size = _pick_tool_size(*safe.size)
        image_url = _to_data_url_png(safe)

        # If false, deps are wrong on Vercel
        if not hasattr(client, "responses"):
            raise HTTPException(
                status_code=500,
                detail="OpenAI SDK too old on this deployment (no client.responses). Ensure requirements.txt is at repo root and redeploy with Clear Cache.",
            )

        # FORCE EDIT: action="edit" + image in context
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

        call0 = calls[0]
        edited = _decode_b64_image(call0.result)

        # back to exact original size
        edited = edited.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

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

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
