from PIL import ImageEnhance, ImageOps
import numpy as np
from PIL import Image

def enhance_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")

    # very gentle global grade
    img = ImageEnhance.Color(img).enhance(1.06)      # vibrance-ish
    img = ImageEnhance.Contrast(img).enhance(1.05)   # midtone pop
    img = ImageEnhance.Brightness(img).enhance(1.01) # keep natural

    return np.array(img)
