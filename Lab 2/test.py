from main import perceptron_gauss_train, get_params
import numpy as np


# testing for positive variance in 1d
def test_perceptron_gauss_train_1d():
    train0 = np.random.uniform(-1, 0, (10, 1))
    train1 = np.random.uniform(0, 1, (10, 1))
    a = perceptron_gauss_train(train0, train1, 1)
    _, sigma, _ = get_params(a, 1)
    assert sigma > 0


# testing positive definiteness in 2d
def test_perceptron_gauss_train_2d():
    train0 = np.random.uniform(-1, 0, (10, 2))
    train1 = np.random.uniform(0, 1, (10, 2))
    a = perceptron_gauss_train(train0, train1, 2)
    _, sigma, _ = get_params(a, 2)
    eig_val, _ = np.linalg.eig(sigma)
    assert (eig_val > 0).all()

