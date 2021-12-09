import numpy as np
import json


def load(file):
    sudoku = np.array(json.load(file)).reshape(81)
    return sudoku
    # # rearranging sudoku into arrays of neighbours for each element
    # neighbours = np.zeros(81, 20)
    # # i don't know how to do this without loops :'(
    # for i in range(81):
    #     # neighbours[i, :(i-1)%9] = sudoku[(i-1)//9, :(i-1)%9]
    #     # neighbours[i, (i-1)%9:8] = sudoku[(i-1)//9, (i-1)%9+1:]
    #     # neighbours[i, 8:8+(i-1)//9] = sudoku[:(i-1)//9, (i-1)%9]
    #     # neighbours[i, 8+(i-1)//9:16] = sudoku[(i-1)//9:, (i-1)%9]
    #     sudoku_unfold = sudoku.reshape(1, 81)
    #     row = i//9
    #     col = i%9
    #     neighbours[i, :8] =


def neighbours_idx(idx):
    neighb_idx = np.arange(0, 20, 1)
    return neighb_idx


def solve(sudoku):
    states = np.array([np.arange(1, 10, 1)])
    objects = np.repeat(states, 81, 0)
    start_values_idx = sudoku.nonzero()
    objects[start_values_idx] = 0
    objects[start_values_idx, sudoku[start_values_idx] - 1] = sudoku[start_values_idx]
    for i in range(objects.shape[0]):
        for j in objects[i].nonzero():
            neighb_idx = neighbours_idx(i)
            flag = np.logical_and(objects[neighb_idx] != 0, objects[neighb_idx] != objects[i, j]).any(axis=1)
            if not flag.all():
                objects[i, j] = 0
    pass


if __name__ == '__main__':
    file = open('sudoku_01.json')
    sudoku = load(file)
    solve(sudoku)
