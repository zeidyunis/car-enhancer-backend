import numpy as np
from PIL import Image, ImageEnhance

def enhance_image(img: Image.Image, strength: float = 0.7) -> np.ndarray:
    img = img.convert("RGB")
    strength = max(0.0, min(1.0, float(strength)))

    img = ImageEnhance.Color(img).enhance(1.05 + 0.08 * strength)
    img = ImageEnhance.Contrast(img).enhance(1.04 + 0.10 * strength)
    img = ImageEnhance.Brightness(img).enhance(1.00 + 0.02 * strength)
    img = ImageEnhance.Sharpness(img).enhance(1.03 + 0.10 * strength)

    return np.array(img)
