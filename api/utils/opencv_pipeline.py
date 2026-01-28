import cv2
import numpy as np
from PIL import Image


def enhance_image(pil_img: Image.Image) -> np.ndarray:
    """
    Deterministic car-photo cleanup:
    - mild lens correction placeholder
    - highlight compression (reduce reflections)
    - gentle contrast
    - NO texture rewriting
    """

    img = np.array(pil_img).astype(np.float32) / 255.0

    # Convert to HSV for highlight control
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)

    # --- 1. Reduce harsh reflections (highlight compression) ---
    # Anything above this is considered glare
    HIGHLIGHT_THR = 0.82

    mask = v > HIGHLIGHT_THR

    # Soft roll-off
    v[mask] = HIGHLIGHT_THR + (v[mask] - HIGHLIGHT_THR) * 0.4

    # --- 2. Restore midtone contrast (avoid flat/matte) ---
    v = np.clip(v, 0, 1)
    v = np.power(v, 0.95)  # slight gamma lift

    # --- 3. Slight saturation recovery (after highlight cut) ---
    s = np.clip(s * 1.04, 0, 1)

    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # --- 4. Gentle micro-contrast (no halos) ---
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.equalizeHist((l * 255).astype(np.uint8)).astype(np.float32) / 255.0
    l = l * 0.7 + lab[:, :, 0] * 0.3

    lab[:, :, 0] = l
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return np.clip(img * 255, 0, 255).astype(np.uint8)
