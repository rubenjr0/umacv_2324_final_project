import cv2
import rerun as rr
from vision import _fix_perspective, _get_face_colors, _preprocess
from colors import _gamma_correction
from solver import solve


rr.init("rerun_example_demo", spawn=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame = _preprocess(frame)
    frame = _gamma_correction(frame, gamma=1)
    rr.log("image/rgb", rr.Image(frame))

    try:
        warped = _fix_perspective(frame, verbose=True)
        colors = _get_face_colors(warped, verbose=True)
        action = solve(colors)
        print(action)
    except Exception as _e:
        continue
