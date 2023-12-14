import cv2
import numpy as np
import rerun as rr

# Note, the ranges are dependent on the lighting conditions
# red, orange and yellow can be very similar

HSV_RANGES = {
    "red": ((0, 60, 100), (9, 255, 255)),
    "orange": ((10, 60, 100), (28, 255, 255)),
    "yellow": ((29, 60, 100), (35, 255, 255)),
    "green": ((36, 70, 100), (85, 255, 255)),
    "blue": ((85, 100, 80), (135, 255, 255)),
    "white": ((0, 0, 200), (255, 60, 255)),
}

def _gamma_correction(channel: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(channel, table)

def _get_hsv(rgb_image: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Convert an RGB image to HSV

    Args:
        rgb_image (np.ndarray): An RGB image

    Returns:
        np.ndarray: An HSV image
    """

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = hsv[..., 0] % 165
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

