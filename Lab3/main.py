import numpy as np
import json


def load(file):
    sudoku = np.array(json.load(file))
    states = np.array([np.arange(1, 10, 1)])
    # create list of objects containing all possible states
    objects = np.repeat(states, 81, 0)
    # find indices of nonzero elements in initial sudoku
    start_values_idx = sudoku.reshape(81).nonzero()
    objects[start_values_idx] = 0
    objects[start_values_idx, sudoku.reshape(81)[start_values_idx] - 1] = \
        sudoku.reshape(81)[start_values_idx]
    return sudoku, objects


# finds indices of neighbours for a specific object
def neighbours_idx(idx):
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


def AC(objects):
    is_changed = 1
    while is_changed:
        is_changed = 0
        for i in range(objects.shape[0]):
            neighb_idx = neighbours_idx(i)
            for j in objects[i].nonzero()[0]:
                flag = np.logical_and(objects[neighb_idx] != 0, objects[neighb_idx] != objects[i, j]).any(axis=1)
                if not flag.all():
                    objects[i, j] = 0
                    is_changed = 1
    return objects


def find_solution(objects):
    objects = AC(objects)
    inv = 1
    while inv:
        i = np.argmax(np.count_nonzero(objects, axis=1) > 1)
        if i:
            iterable = objects[i].nonzero()[0]
            for j in iterable:
                temp = np.copy(objects)
                temp[i] = 0
                temp[i, j] = j + 1
                temp = AC(temp)
                if np.any(temp != 0):
                    objects = np.copy(temp)
                    break
        else:
            if not np.count_nonzero(objects):
                return 'Немає розв\'язку!'
            else:
                return reformat_results(objects)
        # Якщо все викреслено, то задача не є інваріантною
        inv = 0
        return 'Задача не є інваріантною!'


def reformat_results(result):
    solution = result[result != 0].reshape(9, 9)
    return solution


if __name__ == '__main__':
    file = open('sudoku_04.json')
    sudoku, objects = load(file)
    print(find_solution(objects))
