import cv2
import numpy as np
import rerun as rr

# Note, the ranges are dependent on the lighting conditions
# red, orange and yellow can be very similar

HSV_RANGES = {
    "red": ((0, 100, 100), (8, 255, 255)),
    "orange": ((9, 100, 100), (28, 255, 255)),
    "yellow": ((29, 60, 100), (35, 255, 255)),
    "green": ((36, 70, 100), (88, 255, 255)),
    "blue": ((99, 150, 150), (130, 255, 255)),
    "white": ((0, 0, 200), (255, 60, 255)),
}

COLOR_CODES = {
    "red": "R",
    "orange": "O",
    "yellow": "Y",
    "green": "G",
    "blue": "B",
    "white": "W",
}


def _gamma_correction(channel: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(channel, table)

def _white_balance(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    avg_a = np.average(a)
    avg_b = np.average(b)
    a = a - ((avg_a - 128) * (l / 255.0) * 1.2)
    b = b - ((avg_b - 128) * (l / 255.0) * 1.2)
    lab[..., 1] = a
    lab[..., 2] = b
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

def _get_hsv(rgb_image: np.ndarray, verbose: bool = False) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = hsv[..., 0] % 170
    if verbose:
        h, s, v = cv2.split(hsv)
        rr.log("h", rr.Image(h))
        rr.log("s", rr.Image(s))
        rr.log("v", rr.Image(v))
    return hsv


def _classify_color(piece: np.ndarray) -> str:
    """Classify the color of a piece of the cube

    Args:
        piece (np.ndarray): A square piece of the image

    Returns:
        str: The color of the piece (red, orange, yellow, green, blue, white)
    """

    hsv = _get_hsv(piece)
    masks = {name: cv2.inRange(hsv, *ranges) for name, ranges in HSV_RANGES.items()}
    matches = {name: cv2.countNonZero(mask) for name, mask in masks.items()}
    color = max(matches, key=matches.get)
    return color
