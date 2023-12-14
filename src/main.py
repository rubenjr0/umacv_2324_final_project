import cv2
import numpy as np
from colors import _gamma_correction
from solver import solve
from vision import _fix_perspective, _get_face_colors, _preprocess

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame = _preprocess(frame)
    frame = _gamma_correction(frame, gamma=1)

    try:
        cropped, warped, M = _fix_perspective(frame)
        colors = _get_face_colors(cropped)
        action = solve(colors)

        # write action on warped, on (240, 240) from left to right
        cv2.putText(
            warped, str(action), (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4
        )

        # undo perspective transform
        M_inv = np.linalg.inv(M)
        unwarped = cv2.warpPerspective(warped, M_inv, (frame.shape[1], frame.shape[0]))
        frame = cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB)

    except Exception as _e:
        continue
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)
    

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
