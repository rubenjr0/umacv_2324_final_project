import cv2
import imutils
import pathlib
import numpy as np
from imutils import perspective
from matplotlib import pyplot as plt

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


def _load_image(
    path: pathlib.Path, eq: bool = False, verbose: bool = False
) -> np.ndarray:
    img = cv2.imread(path)
    img = imutils.resize(img, width=240)
    if eq:
        img = _eq(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def _threshold_channel(channel, verbose: bool = False):
    blurred = cv2.GaussianBlur(channel, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4
    )
    return adaptive
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    res = cv2.subtract(binarized, adaptive)

    # morph = cv2.morphologyEx(
    #     res, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=4
    # )
    morph = cv2.morphologyEx(
        res, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=6
    )

    if verbose:
        plt.figure()
        plt.subplot(141)
        plt.title("binarized")
        plt.imshow(binarized)
        plt.subplot(142)
        plt.title("adaptive")
        plt.imshow(adaptive)
        plt.subplot(143)
        plt.title("res")
        plt.imshow(res)
        plt.subplot(144)
        plt.title("morphology")
        plt.imshow(morph)

    return adaptive


def _get_square(binarized: np.ndarray):
    edges = imutils.auto_canny(binarized, sigma=0.1)
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
    return max_peri, biggest


def _get_best_channel(a: np.ndarray, b: np.ndarray):
    a_bin = _threshold_channel(a)
    b_bin = _threshold_channel(b)

    a_peri, a_cnts = _get_square(a_bin)
    b_peri, b_cnts = _get_square(b_bin)

    if a_cnts is None and b_cnts is None:
        raise Exception("Both are bad!")
    elif a_cnts is not None and b_cnts is not None:
        if a_peri > b_peri:
            return a_cnts
        else:
            return b_cnts
    elif a_cnts is not None:
        return a_cnts
    else:
        return b_cnts


def _fix_perspective(rgb_img: np.ndarray, verbose: bool = False):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]

    cnts = _get_best_channel(s, v)

    rect = cv2.minAreaRect(cnts)
    box = cv2.boxPoints(rect)
    pts_src = np.array(box, dtype=np.float32)
    origin_x, origin_y = box[0]
    width = rgb_img.shape[0]
    height = rgb_img.shape[1]
    pts_dst = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.float32,
    )
    M, _ = cv2.findHomography(pts_src, pts_dst)
    fixed = cv2.warpPerspective(rgb_img, M, (width, height))

    if verbose:
        plt.figure()
        rgb_cnts = rgb_img.copy()
        cv2.drawContours(rgb_cnts, cnts, -1, (0, 255, 0), 1)
        cv2.drawContours(rgb_cnts, [box.astype(np.int32)], -1, (0, 0, 255), 2)

        plt.subplot(121)
        plt.imshow(rgb_cnts)
        plt.subplot(122)
        plt.imshow(fixed)
        plt.tight_layout()

    corrected = perspective.four_point_transform(rgb_img, box)
    return corrected, M


def _segment_colors(warped: np.ndarray, boost: bool = False, verbose: bool = False):
    if boost:
        warped = cv2.convertScaleAbs(warped, alpha=1.4, beta=0)
        warped = _eq(warped)
    flattened = warped.reshape((-1, 3))
    flattened = np.float32(flattened)
    iters = 16
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 0.1)
    _, labels, centers = cv2.kmeans(
        flattened, 8, None, criteria, iters, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    labels = labels.flatten()
    flattened = centers[labels]
    segmented = flattened.reshape(warped.shape)

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
