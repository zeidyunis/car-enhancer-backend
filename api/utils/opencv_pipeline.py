import numpy as np
from PIL import Image, ImageEnhance, ImageOps

def enhance_image(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")

    # ONLY global, soft corrections (no edge shift)
    img = ImageEnhance.Brightness(img).enhance(1.005)
    img = ImageEnhance.Contrast(img).enhance(1.01)
    img = ImageEnhance.Color(img).enhance(1.005)

    return np.array(img)
