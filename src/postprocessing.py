import cv2
import numpy as np


def postprocess_mask(prob_mask, threshold=0.4, min_area=50):
    binary = (prob_mask > threshold).astype(np.uint8)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = prob_mask.shape
    final_mask = np.zeros((h, w), dtype=np.uint8)
    polygons = []
    max_area = 0.9 * h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            cv2.fillPoly(final_mask, [cnt], 1)
            epsilon = 1.5
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                polygons.append(approx)

    return final_mask, polygons


def get_vehicle_length_width(corners):
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    side1 = (sides[0] + sides[2]) / 2
    side2 = (sides[1] + sides[3]) / 2
    return max(side1, side2), min(side1, side2)
