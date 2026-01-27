import numpy as np
import cv2


def detect_wheels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=50,
        param2=30,
        minRadius=40,
        maxRadius=200
    )

    wheels = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            wheels.append((x, y, r))

    return wheels


def detect_headlights(img):
    h, w = img.shape[:2]

    # heuristic: headlights usually in upper front corners
    boxes = [
        (int(w * 0.05), int(h * 0.25), int(w * 0.3), int(h * 0.5)),
        (int(w * 0.7), int(h * 0.25), int(w * 0.95), int(h * 0.5)),
    ]

    return boxes


def lock_regions(original, enhanced):

    result = enhanced.copy()

    wheels = detect_wheels(original)
    lights = detect_headlights(original)

    # restore wheels
    for x, y, r in wheels:
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        result[mask == 255] = original[mask == 255]

    # restore headlights
    for x1, y1, x2, y2 in lights:
        result[y1:y2, x1:x2] = original[y1:y2, x1:x2]

    return result


def enhance_image(img):

    img = np.array(img)

    # light deterministic enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced, img, lock_regions
