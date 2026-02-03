import numpy as np
from PIL import Image, ImageEnhance


def enhance_image(img: Image.Image, strength: float = 0.55) -> np.ndarray:
    """
    Safe GLOBAL pre-grade. Keep it subtle to avoid hallucinations.
    strength: 0..1
    """
    strength = max(0.0, min(1.0, float(strength)))
    img = img.convert("RGB")

    # Mild but noticeable listing look (global only)
    img = ImageEnhance.Color(img).enhance(1.03 + 0.06 * strength)
    img = ImageEnhance.Contrast(img).enhance(1.03 + 0.10 * strength)
    img = ImageEnhance.Brightness(img).enhance(1.00 + 0.02 * strength)
    img = ImageEnhance.Sharpness(img).enhance(1.02 + 0.06 * strength)  # keep LOW

    return np.array(img)
