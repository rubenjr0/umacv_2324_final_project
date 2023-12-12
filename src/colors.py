import cv2
import numpy as np
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


def _binarize_channel(
    channel, inv: bool = False, name: str = None, verbose: bool = False
):
    # blurred = cv2.medianBlur(blurred, 11)
    blurred = cv2.GaussianBlur(channel, (7, 7), 0)
    sharpened = cv2.addWeighted(channel, 1.5, blurred, -0.5, 0)

    _, sharp_binarized = cv2.threshold(sharpened, 144, 255, cv2.THRESH_BINARY)

    _, blurred_binarized = cv2.threshold(blurred, 144, 255, cv2.THRESH_BINARY)

    binarized = cv2.bitwise_and(blurred_binarized, sharp_binarized)

    if verbose:
        rr.log(f"{name}/blurred", rr.Image(blurred))
        rr.log(f"{name}/sharpened", rr.Image(sharpened))
        rr.log(f"{name}/blurred_binarized", rr.Image(blurred_binarized))
        rr.log(f"{name}/sharp_binarized", rr.Image(sharp_binarized))
        rr.log(f"{name}/binarized", rr.Image(binarized))

    return binarized


def _binarize_hsv_v(rgb_image: np.ndarray):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    _, _, v = cv2.split(hsv)
    binarized = _binarize_channel(v)
    return binarized


def _binarize_lab_ab(rgb_image: np.ndarray):
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    _, a, b = cv2.split(lab)
    a_binarized = _binarize_channel(a)
    b_binarized = _binarize_channel(b)
    return a_binarized, b_binarized


def _binarize_ycrcb_crcb(rgb_image: np.ndarray):
    ycrcb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    cr_binarized = _binarize_channel(cr)
    cb_binarized = _binarize_channel(cb)
    return cr_binarized, cb_binarized


def binarize(rgb_image: np.ndarray, verbose: bool = False):
    req = rgb_image.shape[0] * rgb_image.shape[1] * 0.1
    v = _binarize_hsv_v(rgb_image)
    a, b = _binarize_lab_ab(rgb_image)
    cr, cb = _binarize_ycrcb_crcb(rgb_image)
    channels = [a, b, cr, cb]

    binarized = np.zeros_like(v)
    for channel in channels:
        binarized[channel == 255] += 1
    if np.max(binarized) < 2:
        binarized[v == 255] += 1
    binarized = np.where(binarized >= 2, 255, 0).astype(np.uint8)

    if np.sum(binarized == 255) < req:
        binarized = v

    if verbose:
        rr.log("binarized", rr.Image(binarized))

    return binarized


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))

def _classify_color(piece: np.ndarray) -> (str, float):
    scores = {color: np.median([_distance(piece, shade) for shade in COLORS[color]]) for color in COLORS}
    best_color = max(scores, key=scores.get)
    return best_color, scores[best_color]



