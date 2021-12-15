import numpy as np
from cv2 import imread, imshow, waitKey
import time


def norm_1(x):
    return np.sum(x, axis=len(x.shape)-1)


def abomination(q, g):
    f = np.zeros((g.shape[2]+1, q.shape[0], 1000))
    for i in range(2, g.shape[2]+1):
        f[-i] = np.min(f[-i+1] + q[:, -i+1] + g[:, :, -i+1], axis=0)
    pass


def combine(imgs, masks, alpha, beta):
    m = imgs.shape[0]
    h, w = imgs.shape[1], imgs.shape[2]
    q = alpha * (1 - masks)
    g = np.zeros((m, m, w-1))
    g = beta*(norm_1(imgs[:, np.newaxis, :-1, :] - imgs[np.newaxis, :, 1:, :].repeat(m, axis=0)))
    start_time = time.time()
    abomination(q, g)
    print("--- %s seconds ---" % (time.time() - start_time))
    pass


if __name__ == "__main__":
    imgs = np.zeros((5, 1000, 1000, 3), dtype='uint8')
    for i in range(1, 6):
        imgs[i-1] = imread('image_0{}.png'.format(i))
    masks = np.zeros((5, 1000, 1000, 3), dtype='uint8')
    for i in range(1, 6):
        masks[i-1] = imread('mask_0{}.png'.format(i))

    combine(imgs, masks[..., 0]/255, 1, 1)
