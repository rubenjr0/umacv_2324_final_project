import pathlib

import cv2
import imutils
import numpy as np
import rerun as rr
from colors import HSV_RANGES, _classify_color, _get_hsv
from imutils import perspective

KERNEL_SML = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
KERNEL_MED = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
KERNEL_BIG = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

WIDTH = 480


def _preprocess(bgr_img: np.ndarray):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = imutils.resize(rgb_img, width=WIDTH)

    return rgb_img


def _load_image(path: pathlib.Path) -> np.ndarray:
    img = cv2.imread(path)
    img = _preprocess(img)
    return img


def _morph(stuff: np.ndarray, verbose: bool = False):
    eroded_sml = cv2.erode(stuff, KERNEL_SML, iterations=4)
    eroded_med = cv2.erode(eroded_sml, KERNEL_MED, iterations=1)
    eroded_big = cv2.erode(eroded_med, KERNEL_BIG, iterations=1)

    filtered = cv2.boxFilter(eroded_big, -1, (7, 7), normalize=False)
    dilated = cv2.dilate(filtered, KERNEL_MED, iterations=4)

    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, KERNEL_MED, iterations=4)

    if verbose:
        rr.log("morph/stuff", rr.Image(stuff))
        rr.log("morph/closed", rr.SegmentationImage(closed))

    return closed


def _segment_cube(rgb_img: np.ndarray, verbose: bool = False):
    hsv = _get_hsv(rgb_img)
    mask = np.zeros_like(hsv[:, :, 0])
    for color, (lower, upper) in HSV_RANGES.items():
        color_mask = cv2.inRange(hsv, lower, upper)
        mask |= color_mask
    mask = _morph(mask, verbose=verbose)
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


def _get_box(binarized: np.ndarray, verbose: bool = False):
    edges = cv2.Laplacian(binarized, cv2.CV_8U, ksize=5)
    if verbose:
        rr.log("edges", rr.Image(edges))
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c) * _squareness_error(c))

    max_peri = -np.inf
    biggest = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) == 4:
            if peri > max_peri:
                max_peri = peri
                biggest = approx
                biggest = np.array(sorted(biggest, key=lambda x: x[0][0] + x[0][1]))
    return biggest, max_peri


def _fix_perspective(rgb_img: np.ndarray, verbose: bool = False):
    mask = _segment_cube(rgb_img, verbose=verbose)
    contour, peri = _get_box(mask, verbose=verbose)

    if contour is None:
        raise Exception("No contour found!")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    # if the perimeter is too small the contour is probably noise
    if peri < 250 or peri > 800:
        raise Exception("Contour could be noise!")

    cropped = perspective.four_point_transform(rgb_img, box)

    pts_dst = np.array(
        [[WIDTH, WIDTH], [WIDTH * 2, WIDTH], [WIDTH * 2, WIDTH * 2], [WIDTH, WIDTH * 2]]
    )

    M, _ = cv2.findHomography(box, pts_dst)
    clone = rgb_img.copy()
    # draw rect over clone
    cv2.drawContours(clone, [box.astype("int")], -1, (0, 255, 0), 2)
    warped = cv2.warpPerspective(clone, M, (WIDTH * 3, WIDTH * 3))

    if verbose:
        rr.log("image/corners", rr.Points2D(positions=box, colors=(255, 0, 0), radii=6, labels='face corner'))
        rr.log("warped", rr.Image(warped))

    return cropped, warped, M


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
    colors = np.array(colors).reshape(3, 3)
    return colors, centers
