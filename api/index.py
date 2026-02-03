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
- Do NOT reinterpret or redraw edges, patterns, or textures. Preserve all geometry exactly.
- Do NOT add features that are not present (e.g., headlight washers, sensors, vents, badges, chrome accents).
- If a feature is not visible in the original, it must remain absent.
- Do not “upgrade” the car trim/package.

EDITS MUST BE GLOBAL ONLY (uniform across whole image).
Lens distortion: subtle global de-warp only if needed.
Photorealistic. High quality.
""".strip()

def _client() -> OpenAI:
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


def _guess_out_fmt(upload: UploadFile, pil_format: str | None) -> tuple[str, str, str]:
    ct = (upload.content_type or "").lower()
    pf = (pil_format or "").upper()
    if "png" in ct or pf == "PNG":
        return "PNG", "image/png", "png"
    if "jpeg" in ct or "jpg" in ct or pf in ("JPEG", "JPG"):
        return "JPEG", "image/jpeg", "jpg"
    return "JPEG", "image/jpeg", "jpg"


def _choose_api_canvas(w: int, h: int) -> tuple[int, int]:
    # Choose closest supported canvas by aspect ratio (min letterbox/weirdness)
    target = w / h
    options = [(1024, 1024), (1536, 1024), (1024, 1536)]
    return min(options, key=lambda wh: abs((wh[0] / wh[1]) - target))


def _pad_to_canvas_no_distort(img: Image.Image, canvas_w: int, canvas_h: int):
    # Letterbox without distortion, no crop.
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


def _call_ai_edit(det_path: str, orig_path: str, size_str: str) -> Image.Image:
    """
    Two-image anchoring to reduce hallucinations:
      image[0] = deterministic graded image (what we want)
      image[1] = original canvas (what must be preserved)
    """
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
    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.get("/")
def home():
    # Webpage so visiting the Vercel link doesn't "download"
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Car Enhancer</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; max-width: 820px; margin: 0 auto; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; }
    label { display:block; margin: 12px 0 6px; }
    input[type="file"], input[type="number"] { width: 100%; }
    button { margin-top: 14px; padding: 10px 14px; border-radius: 10px; border: 1px solid #111; background: #111; color: #fff; cursor: pointer; }
    img { max-width: 100%; border-radius: 12px; border: 1px solid #ddd; margin-top: 16px; }
    small { color:#555; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Car Enhancer</h1>
  <p><small>Upload a photo → see the enhanced result. (If OPENAI_API_KEY is missing, enhancement will error.)</small></p>

  <div class="card">
    <form id="f">
      <label>Photo</label>
      <input name="file" type="file" accept="image/*" required />

      <label>Strength (0.0–1.0)</label>
      <input name="strength" type="number" min="0" max="1" step="0.05" value="0.55" />

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
        let t = "";
        try { t = JSON.stringify(await res.json(), null, 2); } catch { t = await res.text(); }
        out.innerHTML = "<pre style='color:#b00;'></pre>";
        out.querySelector("pre").textContent = t;
        return;
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      out.innerHTML = `
        <h3>Result</h3>
        <img src="${url}" />
        <p><a href="${url}" target="_blank" rel="noopener">Open in new tab</a></p>
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


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    strength: float = Form(0.55),
):
    tmp_paths: list[str] = []
    try:
        raw = await file.read()
        pil_in = Image.open(io.BytesIO(raw))
        pil_in = ImageOps.exif_transpose(pil_in)
        original_full = pil_in.convert("RGB")

        out_fmt, out_mime, out_ext = _guess_out_fmt(file, pil_in.format)

        orig_w, orig_h = original_full.size

        # Choose best canvas for original ratio
        canvas_w, canvas_h = _choose_api_canvas(orig_w, orig_h)
        size_str = f"{canvas_w}x{canvas_h}"

        # Letterbox to canvas (no crop, no squish)
        canvas_img, box = _pad_to_canvas_no_distort(original_full, canvas_w, canvas_h)

        # Deterministic pregrade (global only)
        det_np = enhance_image(canvas_img, strength=float(strength)).astype(np.uint8)
        det_img = Image.fromarray(det_np, mode="RGB")

        # Write BOTH for anchoring
        det_path = _write_temp_png(det_img)
        orig_path = _write_temp_png(canvas_img)
        tmp_paths.extend([det_path, orig_path])

        # AI edit with two-image input
        ai_canvas = _call_ai_edit(det_path, orig_path, size_str)

        # Crop back to real content region and resize back to original
        ai_cropped = ai_canvas.crop(box)
        final = ai_cropped.resize((orig_w, orig_h), Image.LANCZOS)

        body = _save_bytes(final, out_fmt)

        return Response(
            content=body,
            media_type=out_mime,
            headers={
                "X-AI-USED": "true",
                "X-ORIG": f"{orig_w}x{orig_h}",
                "X-API-CANVAS": size_str,
                "X-BOX": f"{box[0]},{box[1]},{box[2]},{box[3]}",
                "X-STRENGTH": f"{float(strength):.2f}",
                # Important: inline display, not forced download
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
