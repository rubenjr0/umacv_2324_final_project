import cv2
import numpy as np
from colors import COLOR_CODES, _gamma_correction
from rubik.cube import Cube
from rubik.solve import Solver
from rubik.optimize import optimize_moves
from vision import _fix_perspective, _get_face_colors, _preprocess, WIDTH



def _dialog(dst, txt, line: int = 1, is_warped: bool = False):
    base = WIDTH if is_warped else 16
    cv2.putText(
        dst,
        txt,
        (base, base + 16 * (line - (2 if is_warped else 1))),
        cv2.FONT_HERSHEY_SIMPLEX,
        3 if is_warped else 0.5,
         (0, 255, 0) if is_warped else (255, 255, 255),
        4 if is_warped else 1,
    )

def _format_cube(cube: list[np.ndarray]) -> str:
    top = [COLOR_CODES[c] for c in cube[0].flatten()]
    left = cube[1]
    front = cube[2]
    right = cube[3]
    back = cube[4]
    bottom = [COLOR_CODES[c] for c in cube[5].flatten()]

    # stack left, front, right, back horizontally
    lfrb = np.hstack((left, front, right, back))
    print(lfrb)
    lfrb = lfrb.flatten()
    lfrb = [COLOR_CODES[c] for c in lfrb]

    # join top, lfbr, and bottom
    cube = top + lfrb + bottom
    return "".join(cube)


cap = cv2.VideoCapture(0)
cube = []
solver = None
moves = None
saved_faces = []
to_scan = ['top', 'left', 'front', 'right', 'back', 'bottom']
t = 15
did_save = False
snapshot = None
snapshot_face = None
building_cube = True
solving_cube = False
manual_solve = False
move_ptr = 0
next_move = None

while cap.isOpened():
    _, frame = cap.read()
    frame = _preprocess(frame)
    frame = _gamma_correction(frame, gamma=1)
    try:
        cropped, warped, M = _fix_perspective(frame)
        colors, _ = _get_face_colors(cropped)

        face_color = colors[1, 1]

        if manual_solve:
            if face_color == saved_faces[2]:
                _dialog(warped, next_move, is_warped=True)
            else:
                _dialog(frame, 'Please focus on the front face')

        if building_cube and t == 0 and face_color not in saved_faces:
            snapshot = colors
            snapshot_face = face_color

        # undo perspective transform
        M_inv = np.linalg.inv(M)
        frame = cv2.warpPerspective(warped, M_inv, (frame.shape[1], frame.shape[0]))

    except Exception as _e:
        _dialog(frame, next_move, is_warped=False)
        pass

    if building_cube:
        if t == 0:
            t = 15
            did_save = False
        else:
            t -= 1

        
        _dialog(frame, f"Please scan the {to_scan[0]} face")

        if not did_save and snapshot is not None:
            _dialog(frame, f"Detected: {snapshot_face} face", line=2)
            _dialog(frame, "Hold 's' to save", line=3)
            for i in range(3):
                _dialog(frame, str(snapshot[i, :]), line=14+i)

        if cv2.waitKey(1) & 0xFF == ord("s") and snapshot is not None:
            # append list elements instead of list
            cube.append(snapshot)
            saved_faces.append(snapshot_face)
            to_scan.pop(0)
            snapshot = None
            if len(saved_faces) == 6:
                building_cube = False
                solving_cube = True
                did_save = True
            t = 60
        
        if did_save:
            _dialog(frame, f"Saved: {snapshot_face} face")
            snapshot = None
    elif solving_cube:
        building_cube = False
        solving_cube = False
        manual_solve = True
        cube = _format_cube(cube)
        cube = Cube(cube)
        print('Solving cube...')
        print(cube)
        solver = Solver(cube)
        solver.solve()
        moves = solver.moves
        moves = optimize_moves(moves)
        print('Optimized moves:')
        for move in moves:
            print(move)
        with open('moves.txt', 'w') as f:
            f.write('\n'.join(moves))
    elif manual_solve:
        next_move = moves[move_ptr]
        print('Next move:', next_move)
        _dialog(frame, "Press 'a' to go back, 'd' to go forward")
        if t > 0:
            t -= 1
        if t == 0 and cv2.waitKey(1) & 0xFF == ord("a"):
            if move_ptr > 0:
                move_ptr -= 1
                t = 45
        elif t == 0 and cv2.waitKey(1) & 0xFF == ord("d"):
            if move_ptr < len(moves):
                move_ptr += 1
                t = 45
            else:
                manual_solve = False
                next_move = 'Solved!'
                print('Solved cube!')

        
        

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
