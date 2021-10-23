import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np


if __name__ == '__main__':
    fig, (ax0, ax1) = plt.subplots(1, 2)
    theta0 = np.load('zero.npy')
    theta1 =np.load('one.npy')
    ax0.imshow(theta0)
    ax1.imshow(theta1)
    plt.savefig('results.png')