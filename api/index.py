import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from PIL import Image, ImageOps
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image


APP_VERSION = "chatlike-final-v1"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://be57ce02-6783-4807-a967-7ede7043ec97.lovableproject.com",
        "https://id-preview--be57ce02-6783-4807-a967-7ede7043ec97.lovable.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat-like prompt: SHORT, global-only, no "clarity/sharpen/premium/punchy" language.
PROMPT_CHATLIKE = """
Retouch this exact photo (do not recreate it).

ALLOWED (GLOBAL ONLY, like Lightroom):
- White balance / color cast correction (neutral, natural)
- Small exposure + contrast adjustment (gentle tone curve)
- Mild highlight recovery + mild shadow lift

FORBIDDEN (DO NOT CHANGE):
- Wheels/rims/tires/center caps/logos
- Headlights/taillights/amber markers/reflectors and internal patterns
- Badges/text/plates
- Grille/bumper/trim/sensors/vents/washers
- Body shape, panel gaps, reflections structure
- Background objects/layout
- Do not add/remove anything
- Do not redraw edges or textures

FRAMING:
- No crop/zoom/rotate/aspect change.

IMPORTANT:
- Global adjustments only. No local edits.
- Preserve every physical detail exactly.
""".strip()


def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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


def _choose_api_canvas(w: int, h: int, max_canvas: int = 1024) -> tuple[int, int]:
    """
    Choose a supported output canvas close to aspect ratio.
    Default bias = 1024 canvases for stability.

    max_canvas:
      - 1024 (recommended for least drift)
      - 1536 (if you really want bigger output)
    """
    target = w / h
    if max_canvas <= 1024:
        options = [(1024, 1024), (1024, 1536)]
    else:
        options = [(1024, 1024), (1536, 1024), (1024, 1536)]
    return min(options, key=lambda wh: abs((wh[0] / wh[1]) - target))


def _pad_to_canvas_no_distort(img: Image.Image, canvas_w: int, canvas_h: int):
    """
    Your original behavior: letterbox without distortion, no crop.
    Kept because you said your original code gave better results.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = min(canvas_w / w, canvas_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    fitted = img.resize((new_w, new_h), Image.LANCZOS)

    # Dark border (original behavior)
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


def _call_ai_edit(
    det_path: str,
    orig_path: str,
    size_str: str,
    *,
    anchor_mode: str = "single",
    quality: str = "standard",
) -> Image.Image:
    """
    anchor_mode:
      - "single" (RECOMMENDED): use deterministic image only (min drift)
      - "dual": use [det, orig] (your old mode)

    quality:
      - "standard" (RECOMMENDED): less aggressive, cheaper, usually less drift
      - "high": more aggressive, more drift risk
    """
    client = _client()

    anchor_mode = (anchor_mode or "single").strip().lower()
    quality = (quality or "standard").strip().lower()
    if quality not in ("standard", "high"):
        quality = "standard"

    with open(det_path, "rb") as f_det:
        if anchor_mode == "dual":
            with open(orig_path, "rb") as f_orig:
                result = client.images.edit(
                    model="gpt-image-1.5",
                    image=[f_det, f_orig],
                    prompt=PROMPT_CHATLIKE,
                    size=size_str,
                    quality=quality,
                    output_format="png",
                )
        else:
            # single anchor: det only (closest to "retouch this exact photo")
            result = client.images.edit(
    model="gpt-image-1.5",
    image=[f_det, f_orig],
    prompt=PROMPT,
    size=size_str,
    quality="high",
)


    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.get("/")
def home():
    return HTMLResponse(
        f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Car Enhancer</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; max-width: 860px; margin: 0 auto; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; }}
    label {{ display:block; margin: 12px 0 6px; }}
    input[type="file"], input[type="number"], select {{ width: 100%; }}
    button {{ margin-top: 14px; padding: 10px 14px; border-radius: 10px; border: 1px solid #111; background: #111; color: #fff; cursor: pointer; }}
    img {{ max-width: 100%; border-radius: 12px; border: 1px solid #ddd; margin-top: 16px; }}
    small {{ color:#555; }}
    pre {{ white-space: pre-wrap; }}
    .row {{ display:flex; gap:12px; flex-wrap: wrap; }}
    .row > div {{ flex: 1 1 240px; }}
  </style>
</head>
<body>
  <h1>Car Enhancer</h1>
  <p><small>Version: {APP_VERSION}</small></p>

  <div class="card">
    <form id="f">
      <label>Photo</label>
      <input name="file" type="file" accept="image/*" required />

      <div class="row">
        <div>
          <label>Deterministic Strength (0.0–1.0)</label>
          <input name="strength" type="number" min="0" max="1" step="0.05" value="0.30" />
        </div>
        <div>
          <label>Anchor Mode (drift control)</label>
          <select name="anchor_mode">
            <option value="single" selected>single (recommended)</option>
            <option value="dual">dual (old mode)</option>
          </select>
        </div>
        <div>
          <label>AI Quality</label>
          <select name="ai_quality">
            <option value="standard" selected>standard (recommended)</option>
            <option value="high">high (stronger, more drift risk)</option>
          </select>
        </div>
        <div>
          <label>Max Canvas</label>
          <select name="max_canvas">
            <option value="1024" selected>1024 (recommended)</option>
            <option value="1536">1536 (bigger output)</option>
          </select>
        </div>
      </div>

      <button type="submit">Enhance</button>
    </form>

    <div id="out"></div>
  </div>

  <script>
    const form = document.getElementById('f');
    const out = document.getElementById('out');

    form.addEventListener('submit', async (e) => {{
      e.preventDefault();
      out.innerHTML = "<p>Enhancing…</p>";

      const fd = new FormData(form);
      const res = await fetch('/enhance', {{ method: 'POST', body: fd }});

      if (!res.ok) {{
        let t = "";
        try {{ t = JSON.stringify(await res.json(), null, 2); }} catch {{ t = await res.text(); }}
        out.innerHTML = "<pre style='color:#b00;'></pre>";
        out.querySelector("pre").textContent = t;
        return;
      }}

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      const delta = res.headers.get("X-AI-DELTA") || "";
      const canvas = res.headers.get("X-API-CANVAS") || "";
      const anchor = res.headers.get("X-ANCHOR") || "";
      const q = res.headers.get("X-AI-QUALITY") || "";
      const v = res.headers.get("X-APP-VERSION") || "";

      out.innerHTML = `
        <h3>Result</h3>
        <img src="${{url}}" />
        <p><a href="${{url}}" target="_blank" rel="noopener">Open in new tab</a></p>
        <p><small>Version: ${{v}} | Canvas: ${{canvas}} | Anchor: ${{anchor}} | Quality: ${{q}}${{delta ? (" | AI delta: " + delta) : ""}}</small></p>
      `;
    }});
  </script>
</body>
</html>
        """.strip()
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    strength: float = Form(0.30),

    # knobs (optional; won't break your existing callers)
    anchor_mode: str = Form("single"),
    ai_quality: str = Form("standard"),
    max_canvas: int = Form(1024),
):
    tmp_paths: list[str] = []
    try:
        raw = await file.read()
        pil_in = Image.open(io.BytesIO(raw))
        pil_in = ImageOps.exif_transpose(pil_in)
        original_full = pil_in.convert("RGB")

        out_fmt, out_mime, _ = _guess_out_fmt(file, pil_in.format)

        orig_w, orig_h = original_full.size

        canvas_w, canvas_h = _choose_api_canvas(orig_w, orig_h, max_canvas=int(max_canvas))
        size_str = f"{canvas_w}x{canvas_h}"

        canvas_img, box = _pad_to_canvas_no_distort(original_full, canvas_w, canvas_h)

        strength = float(_clamp(float(strength), 0.0, 1.0))

        # Deterministic pregrade (global only)
        det_np = enhance_image(canvas_img, strength=float(strength)).astype(np.uint8)
        det_img = Image.fromarray(det_np, mode="RGB")

        det_path = _write_temp_png(det_img)
        orig_path = _write_temp_png(canvas_img)
        tmp_paths.extend([det_path, orig_path])

        # AI edit
        ai_canvas = _call_ai_edit(
            det_path,
            orig_path,
            size_str,
            anchor_mode=anchor_mode,
            quality=ai_quality,
        )

        # Debug: quantify how much AI changed pixels on the canvas
        orig_np = np.array(canvas_img, dtype=np.uint8)
        ai_np = np.array(ai_canvas, dtype=np.uint8)
        ai_delta = float(np.mean(np.abs(orig_np.astype(np.int16) - ai_np.astype(np.int16))))

        # Crop back to content + resize back to original
        ai_cropped = ai_canvas.crop(box)
        final = ai_cropped.resize((orig_w, orig_h), Image.LANCZOS)

        body = _save_bytes(final, out_fmt)

        return Response(
            content=body,
            media_type=out_mime,
            headers={
                "X-APP-VERSION": APP_VERSION,
                "X-AI-USED": "true",
                "X-AI-DELTA": f"{ai_delta:.3f}",
                "X-ANCHOR": (anchor_mode or "single").strip().lower(),
                "X-AI-QUALITY": (ai_quality or "standard").strip().lower(),
                "X-ORIG": f"{orig_w}x{orig_h}",
                "X-API-CANVAS": size_str,
                "X-BOX": f"{box[0]},{box[1]},{box[2]},{box[3]}",
                "X-STRENGTH": f"{strength:.2f}",
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
