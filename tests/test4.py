from Lab4.main import combine
import numpy as np


def test_combine_1():
    imgs = np.array([[[[0], [0]], [[0], [0]]], [[[1], [1]], [[1], [1]]]])
    masks = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])
    idx = combine(imgs, masks, 1, 1)
    result = imgs[idx, np.array([[0, 1]]), np.array([[0, 1]])]
    assert np.all(combine(imgs, masks, 10, 1) == np.array([[1, 1], [1, 1]]))


def test_combine_2():
    imgs = np.array([[[[0], [0]], [[0], [0]], [[0], [0]]], [[[1], [1]], [[1], [1]], [[1], [1]]]])
    masks = np.array([[[1, 1], [1, 1], [1, 1]], [[0, 0], [0, 0], [0, 0]]])
    idx = combine(imgs, masks, 1, 1)
    result = imgs[idx, np.array([0, 1, 2])[:, np.newaxis], np.array([0, 1])[np.newaxis, :]]
    assert np.all(combine(imgs, masks, 10, 1) == np.array([[0, 0], [0, 0], [0, 0]]))