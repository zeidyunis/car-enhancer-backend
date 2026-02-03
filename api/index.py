# api/utils/opencv_pipeline.py (FULL REPLACEMENT)
# (Still PIL-only, but tuned to push toward your "after" look BEFORE AI touches it.)
import numpy as np
from PIL import Image, ImageEnhance


def _apply_s_curve(img: Image.Image, strength: float) -> Image.Image:
    """
    Gentle S-curve to deepen blacks + add midtone contrast, while protecting highlights.
    strength: 0..1
    """
    strength = max(0.0, min(1.0, float(strength)))

    # Curve parameters tuned for car listings:
    # Lift shadows slightly but set a deeper black point, and add midtone pop.
    # Implement via LUT on luminance-like channel (approx using per-channel LUT).
    def build_lut():
        lut = []
        for i in range(256):
            x = i / 255.0
            # S-curve: blend between identity and curve depending on strength
            # Curve: y = x + a*(x-0.5)^3  (adds contrast around midtones)
            a = 1.25 * strength
            y = x + a * ((x - 0.5) ** 3) * 4.0
            # Set black point slightly lower and protect highlights
            # pull shadows down a bit + compress top
            y = (y - 0.015 * strength) / (1.0 - 0.01 * strength)
            y = max(0.0, min(1.0, y))
            lut.append(int(round(y * 255.0)))
        return lut

    lut = build_lut()
    r, g, b = img.split()
    r = r.point(lut)
    g = g.point(lut)
    b = b.point(lut)
    return Image.merge("RGB", (r, g, b))


def enhance_image(img: Image.Image, strength: float = 0.70) -> np.ndarray:
    """
    Strong but safe global grade (no object edits):
    - neutralize cast slightly (via color + contrast)
    - deepen blacks + midtone contrast (S-curve)
    - add clarity via mild sharpness (kept low to avoid halos)
    """
    strength = max(0.0, min(1.0, float(strength)))

    img = img.convert("RGB")

    # 1) Slight color (vibrance-ish) boost
    img = ImageEnhance.Color(img).enhance(1.08 + 0.06 * strength)

    # 2) S-curve contrast + deeper blacks
    img = _apply_s_curve(img, strength=strength)

    # 3) Contrast (global)
    img = ImageEnhance.Contrast(img).enhance(1.10 + 0.08 * strength)

    # 4) Tiny brightness correction (keep natural)
    img = ImageEnhance.Brightness(img).enhance(1.01)

    # 5) Controlled clarity/sharpness (small!)
    img = ImageEnhance.Sharpness(img).enhance(1.05 + 0.10 * strength)

    return np.array(img)
