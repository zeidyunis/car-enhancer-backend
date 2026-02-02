import base64
import io
import os
import traceback

from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from PIL import Image, ImageOps
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAIN_MODEL = os.getenv("MAIN_MODEL", "gpt-4.1")
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(4000 * 4000)))  # 16MP


PROMPT = """Enhance this exact photo for an online car sales listing.

CRITICAL RULES (MUST FOLLOW):
- This is an EDIT of the provided photo, NOT a new generated image.
- Do NOT change car identity: model, trim, headlights, grille design, wheels, badges, logos, text, reflections, tint.
- Do NOT add/remove objects or invent chrome trims.
- Preserve ALL readable text/symbols (wheel center caps, brand marks, dashboard buttons/icons).
- Keep geometry stable: no warping, stretching, reshaping.

Allowed adjustments ONLY:
- Neutralize color cast.
- Mild exposure/contrast improvements (natural).
- Mild highlight recovery + shadow lift.
- Subtle clarity/sharpness without halos.
Photorealistic and faithful to the input.
"""


def _load_image(file_bytes: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(file_bytes))
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image upload.")


def _maybe_downscale(im: Image.Image) -> Image.Image:
    w, h = im.size
    if (w * h) <= MAX_PIXELS:
        return im
    scale = (MAX_PIXELS / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _pick_tool_size(w: int, h: int) -> str:
    # tool sizes per docs
    if w > h:
        return "1536x1024"
    if h > w:
        return "1024x1536"
    return "1024x1024"


def _pil_to_data_url_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _decode_image_b64(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.get("/")
def root():
    return {"ok": True}


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload.")

        original = _load_image(data)
        orig_w, orig_h = original.size

        safe = _maybe_downscale(original)
        tool_size = _pick_tool_size(*safe.size)

        img_url = _pil_to_data_url_png(safe)

        # Guard: if SDK is old, fail with clear message
        if not hasattr(client, "responses"):
            raise HTTPException(
                status_code=500,
                detail="Your OpenAI Python SDK is too old (no client.responses). Pin openai==1.63.2 in api/requirements.txt and redeploy.",
            )

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

        calls = [o for o in resp.output if getattr(o, "type", None) == "image_generation_call"]
        if not calls:
            raise HTTPException(status_code=500, detail="No image_generation_call in response.")

        call0 = calls[0]
        edited = _decode_image_b64(call0.result)

        # resize back to original exact dims
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
