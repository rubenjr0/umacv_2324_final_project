import cv2
import numpy as np
import rerun as rr

# (lower, upper) ranges in HSV
HSV_RANGES = {
    "red": ((0, 50, 50), (8, 255, 255)),
    "orange": ((8, 50, 50), (24, 255, 255)),
    "yellow": ((24, 50, 50), (34, 255, 255)),
    "green": ((34, 50, 50), (85, 255, 255)),
    "blue": ((85, 50, 50), (135, 255, 255)),
    "white": ((0, 0, 200), (255, 50, 255)),
}

def _posterize(rgb_image: np.ndarray, level: int = 8) -> np.ndarray:
    indices = np.arange(0, 256)
    divider = np.linspace(0, 255, level + 1)[1]
    quantiz = np.int0(np.linspace(0, 255, level))
    color_levels = np.clip(np.int0(indices / divider), 0, level - 1)
    palette = quantiz[color_levels]
    posterized = palette[rgb_image]
    posterized = cv2.convertScaleAbs(posterized)
    return posterized




def _classify_color(piece: np.ndarray) -> str:
    """Classify the color of a piece of the cube

    Args:
        piece (np.ndarray): A square piece of the image

    Returns:
        str: The color of the piece (red, orange, yellow, green, blue, white)
    """
    hsv = cv2.cvtColor(piece, cv2.COLOR_RGB2HSV)
    
    masks = {name: cv2.inRange(hsv, *ranges) for name, ranges in HSV_RANGES.items()}
    matches = {name: cv2.countNonZero(mask) for name, mask in masks.items()}
    color = max(matches, key=matches.get)
    return color
