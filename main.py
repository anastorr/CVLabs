import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot


def em(train_set, classes):
    # initialising p(k|x) with random values
    n = train_set.shape[0]
    p0 = np.random.uniform(0, 1, n)
    p1 = 1 - p0


if __name__ == '__main__':
    # loading dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    indices = np.argwhere((train_y == 0) | (train_y == 1))
    train_set = np.where(train_X[indices] > 0, 1, 0)

    em(train_set, [0, 1])
    print('hi')
