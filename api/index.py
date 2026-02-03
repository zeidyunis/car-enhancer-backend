import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, Response, JSONResponse
from PIL import Image, ImageOps
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image

app = FastAPI()

PROMPT = """
Edit (not recreate) this exact photo for a premium car sales listing.

GOAL LOOK:
- Neutralize fluorescent/garage color cast (cleaner whites, more neutral).
- Deeper blacks + better midtone contrast with a gentle S-curve (premium, punchy).
- Recover highlights and lift shadows slightly (still natural).
- Add realistic micro-contrast/clarity and clean sharpening (no halos, no HDR).

FRAMING (STRICT):
- Keep the original framing/composition exactly the same.
- Do NOT crop, zoom, rotate, or change aspect ratio.

ABSOLUTE IMMUTABLE (DO NOT CHANGE):
- Wheels/rims/tires/center caps/logos
- Badges/logos/text/plates
- Headlight/taillight shapes and internal patterns
- Body shape, reflections structure, background layout
- Do not add/remove objects
- Do NOT reinterpret edges, patterns, or textures.
- Preserve all geometry exactly.
- Do NOT add features that are not present (e.g., headlight washers, sensors, vents, badges, chrome accents).
- If a feature is not visible in the original, it must remain absent.
- Do not “upgrade” the car trim/package.

EDITS MUST BE GLOBAL ONLY.
Lens distortion: subtle global de-warp only if needed.
Photorealistic. High quality.
""".strip()


def _client() -> OpenAI:
    # IMPORTANT: create client at request-time, not import-time
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def _save_bytes(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "PNG":
        img.save(buf, format="PNG", optimize=True)
    else:
        img.save(buf, format="JPEG", quality=95, subsampling=1, optimize=True)
    return buf.getvalue()


def _choose_api_canvas(w: int, h: int) -> tuple[int, int]:
    ar = w / h
    if ar > 1.15:
        return 1536, 1024
    if ar < 0.87:
        return 1024, 1536
    return 1024, 1024


def _pad_to_canvas_no_distort(img: Image.Image, canvas_w: int, canvas_h: int):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(canvas_w / w, canvas_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    fitted = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    canvas.paste(fitted, (x, y))
    box = (x, y, x + new_w, y + new_h)
    return canvas, box


def _write_temp_png(pil_img: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    path = tmp.name
    tmp.close()
    pil_img.save(path, format="PNG", optimize=True)
    return path


def _call_ai_edit(base_path: str, size_str: str) -> Image.Image:
    client = _client()
    with open(det_path, "rb") as f_det, open(orig_path, "rb") as f_orig:
    result = client.images.edit(
        model="gpt-image-1.5",
        image=[f_det, f_orig],
        prompt=PROMPT,
        size=size_str,
        quality="high",
        output_format="png",
    )

        )
    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.get("/")
def home():
    return HTMLResponse(
        """
<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Car Enhancer</title></head>
<body style="font-family:system-ui;padding:24px;max-width:780px;margin:0 auto;">
<h1>Car Enhancer</h1>
<p>Upload → enhance. If API key missing, enhancement will error but this page will still load.</p>
<form id="f">
  <input name="file" type="file" accept="image/*" required />
  <br/><br/>
  <label>Strength (0–1)</label><br/>
  <input name="strength" type="number" min="0" max="1" step="0.05" value="0.7" />
  <br/><br/>
  <button type="submit">Enhance</button>
</form>
<div id="out"></div>
<script>
const f=document.getElementById('f'), out=document.getElementById('out');
f.addEventListener('submit', async (e)=>{
  e.preventDefault(); out.innerHTML="Enhancing...";
  const fd=new FormData(f);
  const res=await fetch('/enhance',{method:'POST',body:fd});
  if(!res.ok){
    const t=await res.text();
    out.innerHTML="<pre style='white-space:pre-wrap;color:#b00;'></pre>";
    out.querySelector('pre').textContent=t;
    return;
  }
  const blob=await res.blob();
  const url=URL.createObjectURL(blob);
  out.innerHTML=`<img src="${url}" style="max-width:100%;border:1px solid #ddd;border-radius:12px;margin-top:16px;"/>`;
});
</script>
</body></html>
        """.strip()
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/enhance")
async def enhance(file: UploadFile = File(...), strength: float = Form(0.70)):
    tmp_paths = []
    try:
        raw = await file.read()
        pil_in = Image.open(io.BytesIO(raw))
        pil_in = ImageOps.exif_transpose(pil_in)
        original_full = pil_in.convert("RGB")

        orig_w, orig_h = original_full.size
        canvas_w, canvas_h = _choose_api_canvas(orig_w, orig_h)
        size_str = f"{canvas_w}x{canvas_h}"

        canvas_img, box = _pad_to_canvas_no_distort(original_full, canvas_w, canvas_h)

        det_np = enhance_image(canvas_img, strength=float(strength)).astype(np.uint8)
        det_img = Image.fromarray(det_np, mode="RGB")

        base_path = _write_temp_png(det_img)
        tmp_paths.append(base_path)

        ai_canvas = _call_ai_edit(base_path, size_str=size_str)

        ai_cropped = ai_canvas.crop(box)
        final = ai_cropped.resize((orig_w, orig_h), Image.LANCZOS)

        body = _save_bytes(final, "PNG")
        return Response(
            content=body,
            media_type="image/png",
            headers={
                "X-AI-USED": "true",
                "X-ORIG": f"{orig_w}x{orig_h}",
                "X-API-CANVAS": size_str,
                "Content-Disposition": "inline",
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
