import io
import os
import base64
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageEnhance, ImageFilter
from openai import OpenAI

from api.utils.opencv_pipeline import enhance_image


app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"status": "ok"}


PROMPT = """
You are editing an EXISTING real photo for a car sales listing. This is a strict PRESERVATION edit.

PRIMARY GOAL
- Improve the photo’s presentation (cleaner color/exposure/contrast) while keeping the car and scene IDENTICAL.
- Treat this as a “photo correction” task, not a redesign.

ABSOLUTE IMMUTABILITY (MUST NOT CHANGE)

1) Identity & Geometry
- Do not change the car make/model/trim/year appearance.
- Do not change body shape, panel gaps, ride height, stance, proportions, or perspective.
- Do not change wheel size/position, tire profile, or alignment.
- Do not change camera angle, FOV look, or composition.

2) Wheels / Logos / Badges / Text (HIGHEST PRIORITY)
- Do not alter wheels/rims/spokes/tires in any way.
- Do not alter center caps, center-cap logos, lug nuts, valve stems, or tire sidewall text.
- Keep ALL logos crisp and readable.
- Do not alter any badges/emblems/brand marks.
- Do not change or invent any text, numbers, license plates, stickers, decals, or signage.
- Do not blur, warp, or "fix" them.

3) Front/Rear Details
- Do not change grille shape, pattern, or texture.
- Do not change headlights, taillights, DRL shapes, LED patterns, lens textures, or tint.
- Do not change vents, trim pieces, sensors, mirrors, or door handles.

4) Materials & Finish
- Do not add chrome, gloss, sparkle, metallic flakes, or coating effects.
- Do not turn matte into gloss or gloss into matte.
- Do not recolor blacked-out trim, plastics, or rubber.
- Do not add new reflections.

5) Scene Integrity
- Do not add or remove objects.
- Do not change background, sky, ground, or environment.
- Do not invent textures or patterns.

ALLOWED ADJUSTMENTS (GLOBAL ONLY)
Apply ONLY subtle, realistic camera-style corrections:
- White balance correction
- Small exposure adjustment
- Gentle contrast improvement
- Mild highlight recovery
- Mild shadow lift
- Light noise reduction
- Very subtle sharpening (no halos)

STRICT SAFETY RULES
- No local retouching on specific parts.
- No reconstruction.
- If an edit risks changing wheels, logos, headlights, grille, or text, DO NOT apply it.
- Prefer minimal change over detail drift.

OUTPUT QUALITY
- Natural dealer-listing look.
- No stylization.
- No filters.
- No cinematic grading.
- Maintain resolution and aspect ratio.

FINAL SELF-CHECK
Before returning:
- Wheels and logos identical and unwarped
- Headlights and grille unchanged
- All text readable and original
- No new chrome or trim
- No added or removed objects

If any check fails, revert to a more conservative edit.
""".strip()


def _safe_polish(img: Image.Image) -> Image.Image:
    """
    Very mild global polish to reduce "matte" WITHOUT halos/sloppy look.
    """
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Color(img).enhance(1.02)
    img = ImageEnhance.Sharpness(img).enhance(1.04)
    return img


def _resize_to_match(a: np.ndarray, b_img: Image.Image) -> np.ndarray:
    """
    Ensure b matches a's HxW for scoring.
    """
    h, w = a.shape[0], a.shape[1]
    if b_img.size != (w, h):
        b_img = b_img.resize((w, h), Image.LANCZOS)
    return np.array(b_img, dtype=np.uint8)


def _score_similarity(processed_np: np.ndarray, candidate_np: np.ndarray) -> float:
    """
    Lower score = closer to processed (less hallucination).
    Mix of pixel diff + edge diff (no OpenCV, no lockers).
    """
    a = processed_np.astype(np.int16)
    b = candidate_np.astype(np.int16)

    # Mean absolute pixel difference (RGB)
    pix = float(np.mean(np.abs(a - b)))

    # Edge-map difference (structure changes)
    a_edges = Image.fromarray(processed_np).convert("L").filter(ImageFilter.FIND_EDGES)
    b_edges = Image.fromarray(candidate_np).convert("L").filter(ImageFilter.FIND_EDGES)
    ae = np.array(a_edges, dtype=np.int16)
    be = np.array(b_edges, dtype=np.int16)
    edge = float(np.mean(np.abs(ae - be)))

    # Weighted total
    return (pix * 1.0) + (edge * 0.6)


def _call_edit(tmp_path: str) -> bytes:
    """
    One OpenAI edit call -> PNG bytes.
    """
    result = client.images.edit(
        model="gpt-image-1.5",
        image=open(tmp_path, "rb"),
        prompt=PROMPT,
        size="1536x1024",
    )
    out_bytes = base64.b64decode(result.data[0].b64_json)
    return out_bytes


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        # Keep detail but limit huge images
        MAX_SIZE = 2560
        if max(original.size) > MAX_SIZE:
            original.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)

        # Deterministic base (your pipeline)
        processed_np = enhance_image(original).astype(np.uint8)
        processed = Image.fromarray(processed_np, mode="RGB")

        # Save once; we will edit the same deterministic image twice
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        processed.save(tmp_path)

        # ---- BEST OF 2 ----
        out1 = _call_edit(tmp_path)
        out2 = _call_edit(tmp_path)

        img1 = Image.open(io.BytesIO(out1)).convert("RGB")
        img2 = Image.open(io.BytesIO(out2)).convert("RGB")

        cand1_np = _resize_to_match(processed_np, img1)
        cand2_np = _resize_to_match(processed_np, img2)

        s1 = _score_similarity(processed_np, cand1_np)
        s2 = _score_similarity(processed_np, cand2_np)

        chosen_img = img1 if s1 <= s2 else img2
        chosen_score = s1 if s1 <= s2 else s2

        # Mild polish (optional) to reduce matte look
        chosen_img = _safe_polish(chosen_img)

        buf = io.BytesIO()
        chosen_img.save(buf, format="PNG")

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-BO2-S1": str(s1),
                "X-BO2-S2": str(s2),
                "X-BO2-CHOSEN": "1" if s1 <= s2 else "2",
                "X-BO2-SCORE": str(chosen_score),
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )
