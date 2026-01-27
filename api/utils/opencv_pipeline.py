import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def enhance_image(img: Image.Image) -> np.ndarray:
    """
    Lightweight deterministic MVP (no OpenCV):
    - Auto-contrast (gentle)
    - Slight brightness/contrast/color boost
    Returns numpy array RGB.
    """

    if img.mode != "RGB":
        img = img.convert("RGB")

    # gentle autocontrast
    img = ImageOps.autocontrast(img, cutoff=1)

    # subtle adjustments
    img = ImageEnhance.Brightness(img).enhance(1.03)
    img = ImageEnhance.Contrast(img).enhance(1.06)
    img = ImageEnhance.Color(img).enhance(1.04)
    img = ImageEnhance.Sharpness(img).enhance(1.05)

    return np.array(img)
