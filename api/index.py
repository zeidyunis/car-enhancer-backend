# main.py (FULL REPLACEMENT)
import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response, JSONResponse, HTMLResponse
from PIL import Image, ImageOps
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """
Edit (not recreate) this exact photo for a premium car sales listing.

GOAL LOOK:
- Neutralize fluorescent/garage color cast (cleaner whites, more neutral).
- Deeper blacks + better midtone contrast with a gentle S-curve (premium punchy).
- Recover highlights and lift shadows slightly (still natural).
- Add realistic micro-contrast/clarity and clean sharpening (no halos, no HDR).

FRAMING (STRICT):
- Keep the original framing/composition exactly the same.
- Do NOT crop, zoom, rotate, or change aspect ratio.

ABSOLUTE IMMUTABLE (DO NOT CHANGE):
- Car model/shape/panels, reflections structure, background layout
- Wheels/rims/tires/center caps/logos
- Badges/logos/text/plates
- Headlight/taillight shapes and internal patterns
- Do not alter tint level, do not change materials/trim (matte/gloss/chrome)
- Do not add/remove objects

EDITS MUST BE GLOBAL ONLY:
- Apply adjustments uniformly (no local retouching, no object-specific edits).
- No repainting/redrawing.

Lens distortion:
- If there is minor phone lens barrel distortion, apply a subtle global de-warp only.
- If distortion is minimal, do almost nothing.

Photorealistic. High quality.
""".strip()


@app.get("/")
def home():
    # Minimal UI page so your Vercel link "opens a webpage"
    return HTMLResponse(
        """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Car Enhancer</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; max-width: 780px; margin: 0 auto; }
      .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; }
      label { display:block; margin: 12px 0 6px; }
      input[type="file"], input[type="number"] { width: 100%; }
      button { margin-top: 14px; padding: 10px 14px; border-radius: 10px; border: 1px solid #111; background: #111; color: #fff; cursor: pointer; }
      img { max-width: 100%; border-radius: 12px; border: 1px solid #ddd; margin-top: 16px; }
      .row { display:flex; gap:12px; flex-wrap:wrap; }
      .row > div { flex: 1; min-width: 240px; }
      small { color:#555; }
    </style>
  </head>
  <body>
    <h1>Car Enhancer</h1>
    <p><small>Upload an image → get an enhanced version back. Output opens inline (no forced download).</small></p>

    <div class="card">
      <form id="f">
        <label>Photo</label>
        <input name="file" type="file" accept="image/*" required />

        <label>Strength (0.0–1.0)</label>
        <input name="strength" type="number" min="0" max="1" step="0.05" value="0.70" />

        <label>Optional style reference</label>
        <input name="style_ref" type="file" accept="image/*" />

        <button type="submit">Enhance</button>
      </form>

      <div id="out"></div>
    </div>

    <script>
      const form = document.getElementById('f');
      const out = document.getElementById('out');

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        out.innerHTML = "<p>Enhancing…</p>";

        const fd = new FormData(form);
        const res = await fetch('/enhance', { method: 'POST', body: fd });

        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          out.innerHTML = "<pre style='white-space:pre-wrap;color:#b00;'></pre>";
          out.querySelector('pre').textContent = (j.error || "Error") + "\\n\\n" + (j.trace || "");
          return;
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);

        out.innerHTML = `
          <div class="row">
            <div>
              <h3>Result</h3>
              <img src="${url}" />
              <p><a href="${url}" target="_blank" rel="noopener">Open in new tab</a></p>
            </div>
          </div>
        `;
      });
    </script>
  </body>
</html>
        """.strip()
    )


@app.get("/health")
def health():
    return {"status": "ok"}


def _guess_output_format(upload: UploadFile, pil_format: str | None) -> tuple[str, str, str]:
    ct = (upload.content_type or "").lower()
    pf = (pil_format or "").upper()
    if "png" in ct or pf == "PNG":
        return "PNG", "image/png", "png"
    if "jpeg" in ct or "jpg" in ct or pf in ("JPEG", "JPG"):
        return "JPEG", "image/jpeg", "jpg"
    return "JPEG", "image/jpeg", "jpg"


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


def _pad_to_canvas_no_distort(img: Image.Image, canvas_w: int, canvas_h: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
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


def _call_ai_edit(base_path: str, size_str: str, style_ref_path: str | None = None) -> Image.Image:
    kwargs = dict(
        model="gpt-image-1.5",
        prompt=PROMPT,
        size=size_str,
        quality="high",
        output_format="png",
    )

    if style_ref_path:
        with open(base_path, "rb") as f0, open(style_ref_path, "rb") as f1:
            result = client.images.edit(image=[f0, f1], **kwargs)
    else:
        with open(base_path, "rb") as f0:
            result = client.images.edit(image=f0, **kwargs)

    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.post("/enhance")
async def enhance(
    file:
