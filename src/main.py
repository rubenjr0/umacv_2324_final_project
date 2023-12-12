import cv2
import numpy as np
import rerun as rr

from vision import (
    _preprocess,
    _fix_perspective,
    _segment_colors,
    _get_face_colors,
    _threshold_channel,
)
from solver import solve

rr.init("rerun_example_demo", spawn=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame = _preprocess(frame)
    rr.log("image/rgb", rr.Image(frame))
    try:
        fixed = _fix_perspective(frame)
        rr.log("transformed", rr.Image(fixed))
    except Exception as e:
        print(e)
        continue
    segmented = _segment_colors(fixed, boost=True)
    rr.log("transformed/segmented", rr.Image(segmented))
    # colors, centers = _get_face_colors(segmented)
    # action = solve(colors)
    # print(colors)
    # print(action)
