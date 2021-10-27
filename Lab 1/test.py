from main import em
import numpy as np


def test_em():
    assert em(np.array([[[1]], [[1]]])) == (np.array([[1]]), np.array([[1]]))
    assert em(np.array([[[0]], [[0]]])) == (np.array([[0]]), np.array([[0]]))
    theta0, theta1 = em(np.array([[[0.5]], [[0.1]]]))
    assert 0 <= theta0 <= 1 and 0 <= theta1 <= 1
