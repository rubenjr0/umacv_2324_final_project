import cv2
import numpy as np
import rerun as rr

from vision import (
    _eq, _boost,
    _preprocess,
    _fix_perspective,
    _segment_colors,
)
from solver import solve

rr.init("rerun_example_demo", spawn=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame = _preprocess(frame)
    rr.log("image/rgb", rr.Image(frame))
    try:
        warped = _fix_perspective(frame)
        warped = _eq(warped)
        warped = _boost(warped)
        rr.log("warped", rr.Image(warped))
    except Exception as e:
        print(e)
        continue
    # segmented_colors = _segment_colors(warped, boost=True)
    # rr.log("segmented", rr.Image(segmented_colors))
    # colors, centers = _get_face_colors(segmented)
    # action = solve(colors)
    # print(colors)
    # print(action)
