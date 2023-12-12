import cv2
import numpy as np
import rerun as rr

from vision import (
    _preprocess,
    _fix_perspective,
    _segment_colors,
    _get_face_colors
)

rr.init("rerun_example_demo", spawn=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame = _preprocess(frame)
    rr.log("image/rgb", rr.Image(frame))

    try:
        warped = _fix_perspective(frame, verbose=True)
    except Exception as e:
        print(e)
        continue
    # segmented_colors = _segment_colors(warped, boost=True, verbose=True)
    colors = _get_face_colors(warped, verbose=True)
    
    # action = solve(colors)
    # print(action)
