import numpy as np
from enum import Enum


class State(Enum):
    RAND = 1
    CROSS = 2
    CORNERS = 3
    SOLVED = 4


class Action(Enum):
    U = 1
    R = 2
    B = 3
    L = 4
    SOLVED = 5


def missing_for_cross(face: np.ndarray) -> list[(int, int)]:
    target = face[1, 1]
    positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
    missing = [position for position in positions if face[position] != target]
    return missing


def missing_for_corners(face: np.ndarray) -> list[(int, int)]:
    target = face[1, 1]
    positions = [(0, 0), (0, 2), (2, 0), (2, 2)]
    missing = [position for position in positions if face[position] != target]
    return missing


def is_cross(face: np.ndarray) -> bool:
    return len(missing_for_cross(face)) == 0


def is_corners(face: np.ndarray) -> bool:
    return len(missing_for_corners(face)) == 0


def is_solved(face: np.ndarray) -> bool:
    return is_cross(face) and is_corners(face)


def get_state(face: np.ndarray) -> State:
    if is_solved(face):
        return State.SOLVED
    elif is_corners(face):
        return State.CORNERS
    elif is_cross(face):
        return State.CROSS
    else:
        return State.RAND


def solve_cross(face: np.ndarray) -> Action:
    missing = missing_for_cross(face)
    (row, col) = missing[0]
    idx = row * 3 + col
    match idx:
        case 1:
            return Action.U
        case 3:
            return Action.L
        case 5:
            return Action.R
        case 7:
            return Action.B


def solve_corners(face: np.ndarray) -> Action:
    missing = missing_for_corners(face)
    (row, col) = missing[0]
    idx = row * 3 + col
    match idx:
        case 0:
            return Action.U
        case 2:
            return Action.L
        case 6:
            return Action.R
        case 8:
            return Action.B


def solve(face: np.ndarray) -> Action:
    state = get_state(face)
    match state:
        case State.RAND:
            action = solve_cross(face)
        case State.CROSS:
            action = solve_corners(face)
        case State.CORNERS:
            action = None
        case State.SOLVED:
            action = Action.SOLVED
    return action
