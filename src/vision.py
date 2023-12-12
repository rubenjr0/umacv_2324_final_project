import cv2
import imutils
import pathlib
import numpy as np
from imutils import perspective
from matplotlib import pyplot as plt
from colors import binarize, _classify_color

import rerun as rr


def _boost(rgb_img: np.ndarray, verbose: bool = False):
    return cv2.convertScaleAbs(rgb_img, alpha=1.4, beta=0)


def _preprocess(bgr_img: np.ndarray, verbose: bool = False):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = imutils.resize(rgb_img, width=240)
    return rgb_img


def _load_image(path: pathlib.Path, verbose: bool = False) -> np.ndarray:
    img = cv2.imread(path)
    img = _preprocess(img, verbose=verbose)
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


def _segment_cube(rgb_img: np.ndarray, verbose: bool = False):
    binarized = binarize(rgb_img)
    morph = _morph(binarized)
    if verbose:
        rr.log("image/binarized", rr.Image(binarized))
        rr.log("image/mask", rr.Image(morph))
    return morph


def _get_box(binarized: np.ndarray):
    edges = cv2.Laplacian(binarized, cv2.CV_8U, ksize=5)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

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
    segmented = _segment_cube(rgb_img, verbose=verbose)
    contour, peri = _get_box(segmented)

    if contour is None:
        raise Exception("No contour found!")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    # if the perimeter is too small, the contour is probably noise
    rr.log("peri", rr.TimeSeriesScalar(peri))
    if peri < 240:
        raise Exception("Contour too small!")

    warped = perspective.four_point_transform(rgb_img, box)
    if verbose:
        rr.log(
            "image/corners", rr.Points2D(contour[:, 0, :], colors=[255, 0, 0], radii=3)
        )
        rr.log("warped", rr.Image(warped))
    return warped


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
        rr.log("warped/segmented_colors", rr.Image(segmented))

    return segmented


def _get_cells(face: np.ndarray, w_size: int = 10) -> (list[np.ndarray], np.ndarray):
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


def _get_face_colors(
    face: np.ndarray, verbose: bool = False
) -> list[np.ndarray]:
    w_size = face.shape[0] // 9
    cells, centers = _get_cells(face, w_size=w_size)
    face = [_classify_color(cell) for cell in cells]
    colors = [color for (color, _) in face]

    if verbose:
        for i, (center, color) in enumerate(zip(centers, colors)):
            rr.log(f"warped/cell_{i}", rr.Boxes2D(centers=[center], sizes=[w_size, w_size], labels=color))
    
    return colors


def get_face(
    rgb_image: np.ndarray, w_size: int = 10, verbose: bool = False
) -> np.ndarray:
    warped, M = _fix_perspective(rgb_image, verbose=verbose)
    segmented = _segment_colors(warped, boost=True, verbose=verbose)
    colors = _get_face_colors(segmented, w_size=w_size, verbose=verbose)
    return colors
