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


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Add BOTH your lovable preview + production domains here
        "https://be57ce02-6783-4807-a967-7ede7043ec97.lovableproject.com",
        "https://id-preview--be57ce02-6783-4807-a967-7ede7043ec97.lovable.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT = """
Edit (not recreate) this exact photo for a premium car sales listing.

GOAL LOOK:
- Neutralize fluorescent/garage color cast (cleaner whites, more neutral).
- Deeper blacks + better midtone contrast with a gentle S-curve (premium, punchy).
- Recover highlights and lift shadows slightly (still natural).
- Add realistic micro-contrast/clarity and mild sharpening (no halos, no HDR).

FRAMING (STRICT):
- Keep the original framing/composition exactly the same.
- Do NOT crop, zoom, rotate, or change aspect ratio.

ABSOLUTE IMMUTABLE (DO NOT CHANGE):
- DO NOT CHANGE Wheels/rims/tires/center caps/logos
- DO NOT CHANGE Badges/logos/text/plates
- DO NOT CHANGE Headlight/taillight shapes and internal patterns
- DO NOT CHANGE Body shape, reflections structure, background layout
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
        # high quality jpeg (keeps file size reasonable)
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


def _write_temp_png(pil_img: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    path = tmp.name
    tmp.close()
    pil_img.save(path, format="PNG", optimize=True)
    return path


def _maybe_downscale(img: Image.Image, max_side: int) -> tuple[Image.Image, bool]:
    """
    Optional latency/cost guard: downscale huge uploads before running AI.
    NOTE: output will be AI'ed at the smaller size. You can still return the smaller size
    (recommended) OR upscale back (not recommended for sharpness).
    """
    if max_side <= 0:
        return img, False
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img, False
    scale = max_side / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS), True


def _call_ai_edit(det_path: str, orig_path: str) -> Image.Image:
    """
    Two-image anchoring:
      image[0] = deterministic graded image (what we want)
      image[1] = original (what must be preserved)
    """
    client = _client()
    with open(det_path, "rb") as f_det, open(orig_path, "rb") as f_orig:
        result = client.images.edit(
            # Use gpt-image-1.5 (or swap to "chatgpt-image-latest" if you want to test)
            model="gpt-image-1.5",
            image=[f_det, f_orig],
            prompt=PROMPT,
            # IMPORTANT: preserve details/geometry as much as possible
            input_fidelity="high",  # supported on gpt-image-1 / 1.5 :contentReference[oaicite:2]{index=2}
            # IMPORTANT: avoid forced 1024/1536 canvas unless you must
            size="auto",  # supported by the Images API :contentReference[oaicite:3]{index=3}
            quality="high",
            output_format="png",
        )
    out_bytes = base64.b64decode(result.data[0].b64_json)
    return Image.open(io.BytesIO(out_bytes)).convert("RGB")


@app.get("/")
def home():
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
  <p><small>Upload a photo → see the enhanced result.</small></p>

  <div class="card">
    <form id="f">
      <label>Photo</label>
      <input name="file" type="file" accept="image/*" required />

      <label>Strength (0.0–1.0)</label>
      <input name="strength" type="number" min="0" max="1" step="0.05" value="0.30" />

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
    strength: float = Form(0.35),
):
    tmp_paths: list[str] = []
    try:
        raw = await file.read()

        # Decode + fix EXIF rotation
        try:
            pil_in = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                f"Could not decode image. If this is HEIC/HEIF, convert to JPG/PNG before upload. Decoder error: {e}"
            )

        pil_in = ImageOps.exif_transpose(pil_in)
        original = pil_in.convert("RGB")

        out_fmt, out_mime, _ = _guess_out_fmt(file, pil_in.format)

        # Optional: downscale huge images to reduce latency/cost.
        # You can tune with env MAX_SIDE (e.g. 2048 or 2560). Set 0 to disable.
        max_side = int(os.getenv("MAX_SIDE", "2560"))
        working, did_scale = _maybe_downscale(original, max_side=max_side)

        # Deterministic pregrade (global only)
        det_np = enhance_image(working, strength=float(strength)).astype(np.uint8)
        det_img = Image.fromarray(det_np, mode="RGB")

        # Anchor both images
        det_path = _write_temp_png(det_img)
        orig_path = _write_temp_png(working)
        tmp_paths.extend([det_path, orig_path])

        ai_img = _call_ai_edit(det_path, orig_path)

        # IMPORTANT: do NOT upscale back automatically (it looks softer).
        # Return AI image at the working resolution (recommended for listings).
        final = ai_img

        body = _save_bytes(final, out_fmt)

        return Response(
            content=body,
            media_type=out_mime,
            headers={
                "X-AI-USED": "true",
                "X-STRENGTH": f"{float(strength):.2f}",
                "X-DOWNSCALED": "true" if did_scale else "false",
                "X-WORKING": f"{working.size[0]}x{working.size[1]}",
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
