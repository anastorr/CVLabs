import numpy as np
from tensorflow.keras.datasets import mnist


def prob_update(theta0, theta1, x, p):
    product = np.prod((theta1 / theta0) ** x * ((1 - theta1) / (1 - theta0)) ** (1 - x), axis=(1, 2))
    return 1 / (1 + (1 - p) / p * product)


def em(x):
    # initialising p(k|x) with random values
    d = len(x.shape)
    if d > 3:
        x = np.squeeze(x)
    n = x.shape[0]
    p0_x = np.random.uniform(0, 1, n)
    p1_x = 1 - p0_x

    delta = 1
    theta0 = np.zeros(x.shape[1:])
    theta1 = np.zeros(x.shape[1:])

    while delta > 0.001:
        theta0_new = np.sum(x * p0_x[:, np.newaxis, np.newaxis], axis=0) / np.sum(p0_x)
        theta1_new = np.sum(x * p1_x[:, np.newaxis, np.newaxis], axis=0) / np.sum(p1_x)

        p0 = np.sum(p0_x) / n

        p0_x = prob_update(theta0_new, theta1_new, x, p0)
        p1_x = prob_update(theta1_new, theta0_new, x, 1 - p0)

        delta = abs(theta0_new - theta0).max()
        theta0 = theta0_new
        theta1 = theta1_new

    return theta0, theta1


if __name__ == '__main__':
    # loading dataset
    (train_X, train_y), _ = mnist.load_data()
    indices = np.argwhere((train_y == 0) | (train_y == 1))
    train_set = np.where(train_X[indices] > 0, 1, 0)

    theta0, theta1 = em(train_set)
    np.save('zero.npy', theta0)
    np.save('one.npy', theta1)
