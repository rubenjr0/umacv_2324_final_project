import cv2
import imutils
import pathlib
import numpy as np
from imutils import perspective
from matplotlib import pyplot as plt

import rerun as rr


COLORS = {
    "blue": [(0, 0, 104), (0, 255, 0), (151, 151, 255)],
    "green": [(0, 104, 0), (0, 255, 0), (151, 202, 151)],
    "orange": [(104, 68, 0), (255, 165, 0), (255, 218, 151)],
    "red": [(104, 0, 0), (255, 0, 0), (255, 151, 151)],
    "white": [(204, 204, 204), (255, 255, 255), (255, 255, 255)],
    "yellow": [(104, 104, 0), (255, 255, 0), (255, 255, 151)],
}

COLORS = {name: np.array(color) for name, color in COLORS.items()}


def _eq(rgb_img: np.ndarray, verbose: bool = False):
    ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    if verbose:
        plt.figure()
        plt.subplot(121)
        plt.imshow(rgb_img)
        plt.subplot(122)
        plt.imshow(img)
    return img


def _boost(rgb_img: np.ndarray, verbose: bool = False):
    return cv2.convertScaleAbs(rgb_img, alpha=1.4, beta=0)


def _preprocess(bgr_img: np.ndarray, eq: bool = False, verbose: bool = False):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = imutils.resize(rgb_img, width=240)
    if eq:
        rgb_img = _eq(rgb_img)
    return rgb_img


def _load_image(
    path: pathlib.Path, eq: bool = False, verbose: bool = False
) -> np.ndarray:
    img = cv2.imread(path)
    img = _preprocess(img, eq=eq, verbose=verbose)
    if verbose:
        plt.figure()
        plt.imshow(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        plt.figure()
        plt.subplot(131)
        plt.imshow(h)
        plt.subplot(132)
        plt.imshow(s)
        plt.subplot(133)
        plt.imshow(v)
        plt.tight_layout()
    return img


def _morph(stuff: np.ndarray):
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_sml = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    morph = cv2.morphologyEx(stuff, cv2.MORPH_OPEN, kernel_sml, iterations=5)

    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_big, iterations=3)

    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_big, iterations=5)

    return morph


def _binarize_channel(channel, verbose: bool = False):
    blurred = cv2.medianBlur(channel, 11)
    sharpened = cv2.addWeighted(channel, 1.5, blurred, -0.5, 0)
    _, blurred_binarized = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, sharp_binarized = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binarized = cv2.bitwise_and(blurred_binarized, sharp_binarized)

    if verbose:
        rr.log("blurred", rr.Image(blurred))
        rr.log("sharpened", rr.Image(sharpened))
        rr.log("binarized", rr.Image(binarized))

    return binarized


def _segment_cube(s: np.ndarray, v: np.ndarray, verbose: bool = False):
    s_bin = _binarize_channel(s, verbose=verbose)
    v_bin = _binarize_channel(v, verbose=verbose)
    binarized = cv2.bitwise_or(s_bin, v_bin)
    morph = _morph(binarized)
    rr.log("binarized", rr.Image(binarized))
    rr.log("morph", rr.Image(morph))

    return morph


def _get_square(binarized: np.ndarray):
    edges = imutils.auto_canny(binarized, sigma=0.1)
    rr.log("edges", rr.Image(edges))
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    max_peri = -np.inf
    biggest = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            if peri > max_peri:
                max_peri = peri
                biggest = approx
    return biggest


def _fix_perspective(rgb_img: np.ndarray, verbose: bool = False):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]

    segmented = _segment_cube(s, v, verbose=verbose)
    contour = _get_square(segmented)

    if contour is None:
        raise Exception("No contour found!")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    corrected = perspective.four_point_transform(rgb_img, box)
    return corrected


def _segment_colors(warped: np.ndarray, boost: bool = False, verbose: bool = False):
    flattened = warped.reshape((-1, 3))
    flattened = np.float32(flattened)
    iters = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 0.1)
    _, labels, centers = cv2.kmeans(
        flattened, 8, None, criteria, iters, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    labels = labels.flatten()
    flattened = centers[labels]
    segmented = flattened.reshape(warped.shape)
    if boost:
        segmented = _boost(segmented)

    if verbose:
        plt.figure()
        plt.imshow(segmented)

    return segmented


def _get_cells(segmented: np.ndarray, w_size: int = 10) -> list[np.ndarray]:
    height, width, _ = segmented.shape
    cell_width = width // 3
    cell_height = height // 3
    cells = []
    centers = []
    for y in range(3):
        for x in range(3):
            x_min = x * cell_width + w_size
            x_max = x_min + w_size
            y_min = y * cell_height + w_size
            y_max = y_min + w_size
            centers.append((x_min + w_size // 2, y_min + w_size // 2))
            cell = segmented[y_min:y_max, x_min:x_max]
            cells.append(cell)
    return cells, np.array(centers).reshape((3, 3, -1))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))


def _color_distance(color: np.ndarray, reference, margin: int = 50) -> float:
    d_low = _distance(color, np.clip(reference - margin, 0, 255))
    d_base = _distance(color, reference)
    d_upper = _distance(color, np.clip(reference + margin, 0, 255))
    return 3000 / (d_low + d_base + d_upper)


def _classify_color(piece: np.ndarray, colors: dict) -> (str, float):
    scores = {}
    for name, (lower, base, upper) in colors.items():
        score_lower = _color_distance(piece, lower)
        score_base = _color_distance(piece, base)
        score_upper = _color_distance(piece, upper)
        scores[name] = 0.2 * score_lower + 0.5 * score_base + 0.3 * score_upper
    best_color = max(scores, key=scores.get)
    return best_color, scores[best_color]


def _get_face_colors(
    segmented: np.ndarray, w_size: int = 10, verbose: bool = False
) -> list[np.ndarray]:
    cells, centers = _get_cells(segmented, w_size=w_size)
    face = [_classify_color(cell, COLORS) for cell in cells]
    colors = np.array([color for (color, _) in face]).reshape((3, 3))
    if verbose:
        scores = np.array([score for (_, score) in face]).reshape((3, 3))
        plt.figure()
        for y in range(3):
            for x in range(3):
                cell = cells[y * 3 + x]
                color = colors[y, x]
                score = scores[y, x]
                plt.subplot(3, 3, y * 3 + x + 1)
                plt.imshow(cell)
                plt.title(f"{color}: {score:.2f}")
                plt.axis("off")
        plt.tight_layout()
    return colors, centers


def get_face(
    rgb_image: np.ndarray, w_size: int = 10, verbose: bool = False
) -> np.ndarray:
    warped, M = _fix_perspective(rgb_image, verbose=verbose)
    segmented = _segment_colors(warped, boost=True, verbose=verbose)
    colors, centers = _get_face_colors(segmented, w_size=w_size, verbose=verbose)
    return colors, centers
