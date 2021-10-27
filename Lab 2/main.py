import numpy as np
import json


def perceptron_gauss(train0, train1, n):
    m = train0.shape[1]
    a = np.zeros(m)

    train0 = np.concatenate((train0, np.ones((n, m)) * np.NaN), axis=0)
    train1 = np.concatenate((train1, np.ones((n, m)) * np.NaN), axis=0)

    stop = False

    while not stop:
        sigma = np.reshape(a[-n ** 2:], (n, n))
        eig_val, eig_vec = np.linalg.eig(sigma)
        neg_eig_vec = eig_vec[np.where(eig_val <= 0)[0]]
        neg_eig_vec_quadr = np.reshape(np.einsum('...i,...j', neg_eig_vec, neg_eig_vec), (neg_eig_vec.shape[0], n**2))
        constraints = np.concatenate((np.zeros((neg_eig_vec.shape[0], n+1)), neg_eig_vec_quadr), axis=1)

        train0[-n:] = np.concatenate((constraints,
                                      np.ones((n - np.where(eig_val <= 0)[0].size, m)) * np.NaN), axis=0)

        temp0 = np.einsum('ij->i', train0 * a[np.newaxis, :])
        temp1 = np.einsum('ij->i', train1 * a[np.newaxis, :])

        x0 = np.where(temp0 <= 0)[0]
        x1 = np.where(temp1 >= 0)[0]

        if x0.size != 0:
            a += train0[x0[0]]
        elif x1.size != 0:
            a -= train1[x1[0]]
        else:
            stop = True
    return a


def get_params(a, n):
    sigma = np.linalg.inv(np.reshape(a[-n ** 2:], (n, n)))
    mu = np.einsum('i,ij', a[1:n+1], sigma)
    theta = np.e**(-0.5*np.einsum('i,ij,j', mu, sigma, mu) - np.log(2*np.pi*np.linalg.det(sigma)))
    return mu, sigma, theta


if __name__ == '__main__':
    file1 = open('data/train_02.json')
    file2 = open('data/train_01.json')

    train = json.load(file1)
    test = json.load(file2)

    train_0 = np.array(train['outside'])
    train_1 = np.array(train['inside'])

    quadr0 = np.reshape(np.einsum('...i,...j', train_0, train_0), (train_0.shape[0], train_0[0].size ** 2))
    kernel0 = np.concatenate((np.ones((train_0.shape[0], 1)), train_0, quadr0), axis=1)

    quadr1 = np.reshape(np.einsum('...i,...j', train_1, train_1), (train_1.shape[0], train_1[0].size ** 2))
    kernel1 = np.concatenate((np.ones((train_1.shape[0], 1)), train_1, quadr1), axis=1)

    a = perceptron_gauss(kernel0, kernel1, train_0.shape[1])
    params = get_params(a, train_0.shape[1])
