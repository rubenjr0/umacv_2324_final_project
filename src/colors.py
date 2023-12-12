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


def _posterize(rgb_image: np.ndarray, level: int = 8) -> np.ndarray:
    indices = np.arange(0, 256)
    divider = np.linspace(0, 255, level + 1)[1]
    quantiz = np.int0(np.linspace(0, 255, level))
    color_levels = np.clip(np.int0(indices / divider), 0, level - 1)
    palette = quantiz[color_levels]
    posterized = palette[rgb_image]
    posterized = cv2.convertScaleAbs(posterized)
    return posterized


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))


def _classify_color(piece: np.ndarray) -> (str, float):
    scores = {
        color: np.median([_distance(piece, shade) for shade in COLORS[color]])
        for color in COLORS
    }
    best_color = max(scores, key=scores.get)
    return best_color, scores[best_color]
