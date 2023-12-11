import cv2
import numpy as np
import rerun as rr

from vision import _load_image, _fix_perspective, _segment_colors, _get_face_colors
from solver import solve

rr.init("rerun_example_demo", spawn=False)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    _, frame = cap.read()
    cv2.imshow("frame", frame)
    fixed, M = _fix_perspective(frame)
    segmented = _segment_colors(fixed, boost=True)
    colors, centers = _get_face_colors(segmented)
    action = solve(colors)
    print(M)
    print(colors)
    print(action)
    rr.log("image", rr.Image(frame))
    rr.log("transformed", rr.Image(fixed))
    rr.log("transformed/segmented", rr.Image(segmented))
