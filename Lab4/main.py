import numpy as np
from cv2 import imread, imwrite
import time


def norm_1(x):
    return np.sum(abs(x), axis=-1)


def abomination(q, g, h, w, m):
    f = np.zeros((m, h, w))
    f_arg = np.zeros((m, h, w), dtype='int16')
    for i in range(2, w):
        f[..., -i] = np.min(f[..., -i+1] + q[:, :, -i+1] + g[:, :, :, -i+1], axis=1)
        f_arg[..., -i] = np.argmin(f[..., -i+1] + q[:, :, -i+1] + g[:, :, :, -i+1], axis=1)
    # solving for k*
    ans = np.zeros((h, w), dtype='int16')
    ans[..., 0] = np.argmin(q[:, :, 0] + f[..., 1], axis=0)
    idx = np.arange(0, h, 1)
    for i in range(1, w):
        ans[..., i] = f_arg[ans[..., i-1], idx, i]
    return ans


def combine(imgs, masks, alpha, beta):
    m = imgs.shape[0]
    h = imgs.shape[1]
    w = imgs.shape[2]
    q = alpha * (1 - masks)
    g = beta*(norm_1(imgs[:, np.newaxis, :, :-1] - imgs[np.newaxis, :, :, :-1].repeat(m, axis=0))
              + norm_1(imgs[:, np.newaxis, :, 1:] - imgs[np.newaxis, :, :, 1:].repeat(m, axis=0)))
    ans = abomination(q, g, h, w, m)
    return ans


if __name__ == "__main__":
    start_time = time.time()
    h = 1000
    w = 1000
    m = 5
    imgs = np.zeros((m, h, w, 3), dtype='int16')
    for i in range(1, 6):
        imgs[i-1] = imread('image_0{}.png'.format(i))
    masks = np.zeros((m, h, w, 3), dtype='int16')
    for i in range(1, 6):
        masks[i-1] = imread('mask_0{}.png'.format(i))

    k = combine(imgs, masks[..., 0]/255, 1000, 10)
    result = imgs[k, np.arange(0, h, 1)[:, np.newaxis], np.arange(0, w, 1)[np.newaxis, :], :]
    imwrite('result.png', result)
    print("--- %s seconds ---" % (time.time() - start_time))

