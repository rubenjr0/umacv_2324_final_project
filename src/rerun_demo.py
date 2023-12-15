from sys import argv

import cv2
import numpy as np
import rerun as rr
from colors import _gamma_correction
from vision import _fix_perspective, _get_face_colors, _preprocess
from webcolors import name_to_rgb

camera_idx = int(argv[1]) if len(argv) > 1 else -1
gamma = float(argv[2]) if len(argv) > 2 else 0.6

print('Capturing from camera', camera_idx, 'with gamma correction', gamma)

cap = cv2.VideoCapture(camera_idx)

rr.init("rubik", spawn=True)

while cap.isOpened():
    _, frame = cap.read()
    frame = _preprocess(frame)
    rr.log("image/rgb", rr.Image(frame))
    frame = _gamma_correction(frame, gamma=gamma)
    rr.log("image/corrected", rr.Image(frame))

    try:
        cropped, warped, M = _fix_perspective(frame, verbose=True)
        rr.log("face", rr.Image(cropped))

        colors, centers = _get_face_colors(cropped)
        for i, (center, color) in enumerate(zip(centers, colors.flatten())):
            rr.log(
                f"face/cell/{i}",
                rr.Boxes2D(
                    centers=[center],
                    sizes=[10, 10],
                    labels=color,
                    colors=name_to_rgb(color),
                ),
            )

        # undo perspective transform
        M_inv = np.linalg.inv(M)
        frame = cv2.warpPerspective(warped, M_inv, (frame.shape[1], frame.shape[0]))

    except Exception as _e:
        print(_e)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
