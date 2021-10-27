import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import json


if __name__ == '__main__':
    mu = np.load('mu.npy')
    sigma = np.load('sigma.npy')
    theta = np.load('theta.npy')

    file1 = open('data/train_02.json')
    file2 = open('data/train_01.json')

    train = json.load(file1)
    test = json.load(file2)

    train_0 = np.array(train['outside'])
    train_1 = np.array(train['inside'])

    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x, y))

    z = multivariate_normal(mu, sigma)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, z.pdf(pos))
    ax2.contour(x, y, z.pdf(pos), levels=[theta])

    ax2.scatter(train_1[:, 0], train_1[:, 1], c='white')
    ax2.scatter(train_0[:, 0], train_0[:, 1], c='red')

    plt.savefig('plot_train.png')
