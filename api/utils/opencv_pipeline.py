import numpy as np
from PIL import Image, ImageEnhance, ImageOps

def enhance_image(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Brightness(img).enhance(1.02)
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Color(img).enhance(1.03)
    img = ImageEnhance.Sharpness(img).enhance(1.02)

    return np.array(img)
