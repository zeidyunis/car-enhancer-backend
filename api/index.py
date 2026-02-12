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
        "https://be57ce02-6783-4807-a967-7ede7043ec97.lovableproject.com",
        "https://id-preview--be57ce02-6783-4807-a967-7ede7043ec97.lovable.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# PROMPTS (Normal + Micro)
# -------------------------

PROMPT_NORMAL = """
Edit (not recreate) this exact photo for a car sales listing.

GOAL LOOK (GLOBAL ONLY):
- Neutralize color cast (cleaner whites, more neutral).
- Slightly deeper blacks + better midtone contrast with a gentle S-curve.
- Recover highlights and lift shadows slightly (still natural).
- Very subtle clarity only (do NOT invent edges; no HDR; no halos).

FRAMING (STRICT):
- Keep the original framing/composition exactly the same.
- Do NOT crop, zoom, rotate, or change aspect ratio.

ABSOLUTE IMMUTABLE (DO NOT CHANGE):
- DO NOT CHANGE wheels/rims/tires/center caps/logos
- DO NOT CHANGE badges/logos/text/plates
- DO NOT CHANGE headlight/taillight shapes, amber markers/reflectors, and internal patterns
- DO NOT CHANGE body shape, panel gaps, trim pieces, sensors, washers, vents, chrome accents
- Do NOT add/remove objects
- Do NOT redraw or reinterpret edges, patterns, or textures. Preserve all geometry exactly.
- If a feature is not visible in the original, it must remain absent.
- Do not “upgrade” the trim/package.

IMPORTANT:
- Adjust lighting/color only. No object reconstruction. No redesign.
- Prefer subtle, realistic retouching.
""".strip()

PROMPT_MICRO = """
Edit (not recreate) this exact photo with a MICRO retouch only.

ALLOWED (GLOBAL ONLY):
- Very small white balance correction
- Very small exposure/contrast adjustment
- Very small highlight recovery and shadow lift

ABSOLUTE IMMUTABLE (DO NOT CHANGE):
- Do NOT change wheels/rims/tires/center caps/logos
- Do NOT change headlights/taillights/amber markers/reflectors
- Do NOT change badges/grille/bumper shapes or any trim/sensors/washers/vents
- Do NOT add/remove objects
- Do NOT redraw edges or textures
- Do NOT modify geometry, proportions, reflections structure, or background layout

FRAMING (STRICT):
- No crop/zoom/rotate/aspect change.

Return the same photo, only slightly better color and exposure. No sharpening.
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -------------------------
# Anti-hallucination gate
# -------------------------

def _norm_boxes_default() -> list[tuple[float, float, float, float]]:
    """
    Conservative "typical car listing" protected ROIs in normalized coords (x0,y0,x1,y1)
    for a 3/4 front view:
      - rear wheel region (left/lower)
      - front wheel region (mid/right-lower)
      - headlight/front fascia region (right/upper-mid)
    If these are slightly off, the GLOBAL gate still helps; you can tune later.
    """
    return [
        (0.12, 0.58, 0.33, 0.88),  # rear wheel area
        (0.44, 0.56, 0.70, 0.90),  # front wheel area
        (0.58, 0.28, 0.92, 0.62),  # headlight + grille area
    ]


def _boxes_from_norm(norm_boxes: list[tuple[float, float, float, float]], w: int, h: int) -> list[tuple[int, int, int, int]]:
    out = []
    for x0, y0, x1, y1 in norm_boxes:
        bx0 = int(_clamp(x0, 0.0, 1.0) * w)
        by0 = int(_clamp(y0, 0.0, 1.0) * h)
        bx1 = int(_clamp(x1, 0.0, 1.0) * w)
        by1 = int(_clamp(y1, 0.0, 1.0) * h)
        # ensure valid non-empty
        bx0, bx1 = sorted((max(0, bx0), min(w, bx1)))
        by0, by1 = sorted((max(0, by0), min(h, by1)))
        if bx1 - bx0 >= 8 and by1 - by0 >= 8:
            out.append((bx0, by0, bx1, by1))
    return out


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    # a,b uint8 HxWx3
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def _gate_drift(
    orig_canvas: Image.Image,
    ai_canvas: Image.Image,
    roi_threshold: float = 7.0,
    global_threshold: float = 10.0,
) -> tuple[bool, dict]:
    """
    Returns (ok, metrics).
    - roi_threshold: mean absolute diff per ROI (0-255) allowed
    - global_threshold: mean absolute diff over whole image allowed
    """
    orig = np.array(orig_canvas.convert("RGB"), dtype=np.uint8)
    ai = np.array(ai_canvas.convert("RGB"), dtype=np.uint8)

    if orig.shape != ai.shape:
        return False, {"reason": "shape_mismatch", "orig_shape": str(orig.shape), "ai_shape": str(ai.shape)}

    h, w = orig.shape[:2]

    metrics = {}
    metrics["global_mad"] = _mean_abs_diff(orig, ai)

    # ROI checks (wheels/headlights-ish)
    rois = _boxes_from_norm(_norm_boxes_default(), w, h)
    roi_mads = []
    for i, (x0, y0, x1, y1) in enumerate(rois):
        m = _mean_abs_diff(orig[y0:y1, x0:x1], ai[y0:y1, x0:x1])
        roi_mads.append(m)
        metrics[f"roi{i}_mad"] = m
        metrics[f"roi{i}_box"] = f"{x0},{y0},{x1},{y1}"

    metrics["roi_max_mad"] = max(roi_mads) if roi_mads else 0.0

    # Decision
    ok = True
    if metrics["global_mad"] > global_threshold:
        ok = False
        metrics["reason"] = "global_drift"
    if metrics["roi_max_mad"] > roi_threshold:
        ok = False
        metrics["reason"] = "roi_drift"

    return ok, metrics


# -------------------------
# OpenAI call
# -------------------------

def _call_ai_edit(det_path: str, orig_path: str, size_str: str, prompt_text: str) -> Image.Image:
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
            prompt=prompt_text,
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
    input[type="file"], input[type="number"], select { width: 100%; }
    button { margin-top: 14px; padding: 10px 14px; border-radius: 10px; border: 1px solid #111; background: #111; color: #fff; cursor: pointer; }
    img { max-width: 100%; border-radius: 12px; border: 1px solid #ddd; margin-top: 16px; }
    small { color:#555; }
    pre { white-space: pre-wrap; }
    .row { display:flex; gap:12px; flex-wrap: wrap; }
    .row > div { flex: 1 1 220px; }
  </style>
</head>
<body>
  <h1>Car Enhancer</h1>
  <p><small>Upload a photo → see the enhanced result. (If OPENAI_API_KEY is missing, enhancement will error.)</small></p>

  <div class="card">
    <form id="f">
      <label>Photo</label>
      <input name="file" type="file" accept="image/*" required />

      <div class="row">
        <div>
          <label>Deterministic Strength (0.0–1.0)</label>
          <input name="strength" type="number" min="0" max="1" step="0.05" value="0.35" />
        </div>
        <div>
          <label>AI Mode</label>
          <select name="ai_mode">
            <option value="normal" selected>normal</option>
            <option value="micro">micro (safest)</option>
          </select>
        </div>
        <div>
          <label>Drift Gate</label>
          <select name="gate">
            <option value="true" selected>on</option>
            <option value="false">off</option>
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

      const gate = res.headers.get("X-GATE") || "";
      const drift = res.headers.get("X-DRIFT") || "";
      const retry = res.headers.get("X-AI-RETRY") || "";
      const aiUsed = res.headers.get("X-AI-USED") || "";

      out.innerHTML = `
        <h3>Result</h3>
        <img src="${url}" />
        <p><a href="${url}" target="_blank" rel="noopener">Open in new tab</a></p>
        <p><small>
          AI: ${aiUsed} | Gate: ${gate} ${retry ? ("| Retry: " + retry) : ""} ${drift ? ("| Drift: " + drift) : ""}
        </small></p>
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

    # Optional controls (won't break existing clients)
    ai_mode: str = Form("normal"),          # "normal" | "micro"
    gate: str = Form("true"),               # "true" | "false"
    roi_threshold: float = Form(7.0),       # lower = stricter (more rejects)
    global_threshold: float = Form(10.0),   # lower = stricter (more rejects)
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
        strength = float(_clamp(float(strength), 0.0, 1.0))
        det_np = enhance_image(canvas_img, strength=strength).astype(np.uint8)
        det_img = Image.fromarray(det_np, mode="RGB")

        # Write BOTH for anchoring
        det_path = _write_temp_png(det_img)
        orig_path = _write_temp_png(canvas_img)
        tmp_paths.extend([det_path, orig_path])

        # Choose prompt
        ai_mode = (ai_mode or "normal").strip().lower()
        prompt_text = PROMPT_MICRO if ai_mode == "micro" else PROMPT_NORMAL

        # AI edit (attempt 1)
        ai_canvas = _call_ai_edit(det_path, orig_path, size_str, prompt_text=prompt_text)

        # Drift gate (optional) + auto-retry
        gate_on = (gate or "true").strip().lower() == "true"
        drift_metrics = {}
        retried = "false"

        if gate_on:
            ok, drift_metrics = _gate_drift(
                orig_canvas=canvas_img,
                ai_canvas=ai_canvas,
                roi_threshold=float(roi_threshold),
                global_threshold=float(global_threshold),
            )

            if not ok:
                # Retry once: tighten prompt + reduce deterministic strength a bit
                retried = "true"
                strength_retry = _clamp(strength * 0.8, 0.15, 0.35)

                det_np2 = enhance_image(canvas_img, strength=float(strength_retry)).astype(np.uint8)
                det_img2 = Image.fromarray(det_np2, mode="RGB")

                det_path2 = _write_temp_png(det_img2)
                tmp_paths.append(det_path2)

                ai_canvas2 = _call_ai_edit(
                    det_path2,
                    orig_path,
                    size_str,
                    prompt_text=PROMPT_MICRO,  # safest on retry
                )

                ok2, drift_metrics2 = _gate_drift(
                    orig_canvas=canvas_img,
                    ai_canvas=ai_canvas2,
                    roi_threshold=float(roi_threshold),
                    global_threshold=float(global_threshold),
                )

                if ok2:
                    ai_canvas = ai_canvas2
                    drift_metrics = drift_metrics2
                    strength = strength_retry
                    prompt_text = PROMPT_MICRO
                else:
                    # Hard fallback: return deterministic-only (guaranteed no hallucination)
                    ai_canvas = det_img
                    drift_metrics = drift_metrics2
                    prompt_text = "DETERMINISTIC_FALLBACK"

        # Crop back to real content region and resize back to original
        ai_cropped = ai_canvas.crop(box)
        final = ai_cropped.resize((orig_w, orig_h), Image.LANCZOS)

        body = _save_bytes(final, out_fmt)

        # Minimal drift header (keep short)
        drift_reason = drift_metrics.get("reason", "")
        drift_summary = ""
        if drift_reason:
            drift_summary = f"{drift_reason};g={drift_metrics.get('global_mad', 0):.2f};rmax={drift_metrics.get('roi_max_mad', 0):.2f}"

        return Response(
            content=body,
            media_type=out_mime,
            headers={
                "X-AI-USED": "true",
                "X-AI-MODE": ai_mode,
                "X-AI-RETRY": retried,
                "X-GATE": "on" if gate_on else "off",
                "X-DRIFT": drift_summary,
                "X-ORIG": f"{orig_w}x{orig_h}",
                "X-API-CANVAS": size_str,
                "X-BOX": f"{box[0]},{box[1]},{box[2]},{box[3]}",
                "X-STRENGTH": f"{float(strength):.2f}",
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
