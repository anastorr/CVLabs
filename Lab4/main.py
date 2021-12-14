import numpy as np
from cv2 import imread, imshow, waitKey


def combine(imgs, masks, alpha, beta):
    m = imgs.shape[0]
    h, w = imgs.shape[1], imgs.shape[2]
    pass


if __name__ == "__main__":
    imgs = np.zeros((5, 1000, 1000, 3), dtype='uint8')
    for i in range(1, 6):
        imgs[i-1] = imread('image_0{}.png'.format(i))
    masks = np.zeros((5, 1000, 1000, 3), dtype='uint8')
    for i in range(1, 6):
        masks[i-1] = imread('mask_0{}.png'.format(i))
