import pathlib

import cv2
import imutils
import numpy as np
import rerun as rr
from colors import HSV_RANGES, _classify_color, _get_hsv
from imutils import perspective

kernel_xl = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
kernel_med = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_sml = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def _preprocess(bgr_img: np.ndarray, verbose: bool = False):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = imutils.resize(rgb_img, width=240)

    return rgb_img


def _load_image(path: pathlib.Path, verbose: bool = False) -> np.ndarray:
    img = cv2.imread(path)
    img = _preprocess(img, verbose=verbose)
    if verbose:
        rr.log("image/rgb", rr.Image(img))
    return img


def _morph(stuff: np.ndarray):
    morph_sml = cv2.morphologyEx(stuff, cv2.MORPH_OPEN, kernel_sml, iterations=2)
    morph_med = cv2.morphologyEx(morph_sml, cv2.MORPH_CLOSE, kernel_med, iterations=3)
    morph_big = cv2.morphologyEx(morph_med, cv2.MORPH_OPEN, kernel_big, iterations=2)
    return morph_big


def _segment_cube(rgb_img: np.ndarray, verbose: bool = False):
    hsv = _get_hsv(rgb_img, verbose=True)
    mask = np.zeros_like(hsv[:, :, 0])
    for color, (lower, upper) in HSV_RANGES.items():
        color_mask = cv2.inRange(hsv, lower, upper)
        mask |= color_mask
        if verbose:
            rr.log(f"mask/{color}", rr.Image(color_mask))

    mask = _morph(mask)
    if verbose:
        rr.log("morph", rr.Image(mask))

    return mask


def _compactness(contour: np.ndarray) -> float:
    """Calculate the compactness of a contour

    Args:
        contour (np.ndarray): A contour

    Returns:
        float: The compactness of the contour
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    compactness = 4 * np.pi * area / (perimeter**2)
    return compactness


def _squareness_error(contour: np.ndarray) -> float:
    """Calculate the squareness error of a contour

    Args:
        contour (np.ndarray): A contour

    Returns:
        float: The squareness error of the contour
    """

    compactness = _compactness(contour)
    return 1e6 * (1 / 16 - compactness) ** 2


def _get_box(binarized: np.ndarray):
    edges = cv2.Laplacian(binarized, cv2.CV_8U, ksize=5)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=_squareness_error)

    max_peri = -np.inf
    biggest = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) == 4:
            if peri > max_peri:
                max_peri = peri
                biggest = approx
    return biggest, max_peri


def _fix_perspective(rgb_img: np.ndarray, verbose: bool = False):
    segmented = _segment_cube(rgb_img, verbose=True)
    contour, peri = _get_box(segmented)

    if contour is None:
        raise Exception("No contour found!")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    # if the perimeter is too small the contour is probably noise
    if peri < 240:
        raise Exception("Contour too small!")

    warped = perspective.four_point_transform(rgb_img, box)
    if verbose:
        rr.log(
            "image/corners",
            rr.Points2D(contour[:, 0, :], colors=[255, 0, 0], radii=3),
        )
        rr.log("warped", rr.Image(warped))
    return warped


def _get_cells(face: np.ndarray, w_size: int) -> (list[np.ndarray], np.ndarray):
    height, width, _ = face.shape
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
            center = (x_min + x_max) // 2, (y_min + y_max) // 2
            cell = face[y_min:y_max, x_min:x_max]
            cells.append(cell)
            centers.append(center)
    centers = np.array(centers)
    return cells, centers


def _get_face_colors(face: np.ndarray, verbose: bool = False) -> list[np.ndarray]:
    w_size = face.shape[0] // 9
    cells, centers = _get_cells(face, w_size=w_size)
    colors = [_classify_color(cell) for cell in cells]
    if verbose:
        for i, (center, color) in enumerate(zip(centers, colors)):
            rr.log(
                f"warped/cell_{i}",
                rr.Boxes2D(
                    centers=[center],
                    sizes=[w_size*2, w_size*2],
                    labels=color,
                    class_ids=[i],
                ),
            )
    colors = np.array(colors).reshape(3, 3)
    return colors


def get_face(
    rgb_image: np.ndarray, w_size: int = 10, verbose: bool = False
) -> np.ndarray:
    warped = _fix_perspective(rgb_image, verbose=verbose)
    colors = _get_face_colors(warped, w_size=w_size, verbose=verbose)
    return colors
