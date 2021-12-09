from Lab3.main import load, find_solution
import numpy as np


def test_find_solution_1():
    file = open('sudoku_03.json')
    sudoku, objects = load(file)
    assert find_solution(objects) == 'Немає розв\'язку!'


def test_find_solution_2():
    file = open('sudoku_04.json')
    sudoku, objects = load(file)
    assert find_solution(objects) == 'Немає розв\'язку!'