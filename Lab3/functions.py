import numpy as np
import json


def load(file):
    sudoku = np.array(json.load(file))
    return sudoku


def neighbours_idx(idx, sudoku):
    row = idx // 9
    col = idx % 9
    section_row = row // 3
    section_col = col // 3
    indices = np.arange(0, 81, 1).reshape(9, 9)
    section = np.copy(indices[3*section_row:3*section_row+3, 3*section_col:3*section_col+3])
    section[row % 3] = -1
    section[:, col % 3] = -1
    neighb_idx = np.concatenate((indices[row, :col], indices[row, col+1:], indices[:row, col], indices[row+1:, col],
                                 section[section != -1]))
    return neighb_idx


def solve(sudoku):
    states = np.array([np.arange(1, 10, 1)])
    # create list of objects containing all possible states
    objects = np.repeat(states, 81, 0)
    # find indices of nonzero elements in initial sudoku
    start_values_idx = sudoku.reshape(81).nonzero()
    objects[start_values_idx] = 0
    objects[start_values_idx, sudoku.reshape(81)[start_values_idx] - 1] = \
        sudoku.reshape(81)[start_values_idx]
    is_changed = 1
    while is_changed:
        is_changed = 0
        for i in range(objects.shape[0]):
            neighb_idx = neighbours_idx(i, sudoku)
            for j in objects[i].nonzero()[0]:
                flag = np.logical_and(objects[neighb_idx] != 0, objects[neighb_idx] != objects[i, j]).any(axis=1)
                if not flag.all():
                    objects[i, j] = 0
                    is_changed = 1
    return objects


if __name__ == '__main__':
    file = open('sudoku_01.json')
    sudoku = load(file)
    solve(sudoku)
