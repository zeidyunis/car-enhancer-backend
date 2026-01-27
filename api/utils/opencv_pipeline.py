import cv2
import numpy as np


def enhance_image(img):

    # Convert to OpenCV format
    img = np.array(img)

    h, w = img.shape[:2]

    # ---------- Lens Correction ----------
    K = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ])

    D = np.array([-0.15, 0.05, 0, 0])

    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (w, h), cv2.CV_32FC1
    )

    img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # ---------- Perspective Fix ----------
    pts1 = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    pts2 = np.float32([
        [20, 20],
        [w-20, 20],
        [20, h-20],
        [w-20, h-20]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (w, h))

    # ---------- Tone Curve ----------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img
