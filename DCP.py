# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def dark_channel(img, size=30):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img


def get_atmo(img, percent=0.005):
    mean_perpix = np.mean(img, axis=2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)


def get_trans(img, atom, w=0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t


def guided_filter(p, i, r, e):
    # 1
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    # 2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # 3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    # 4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # 5
    q = mean_a * i + mean_b
    return q


def recover(img, t, a):
    r = np.empty_like(img)
    for i in range(3):
        r[:, :, i] = (img[:, :, i] - a) / t + a
    return r


def DCP_dehaze(im):
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    atmo = get_atmo(img)
    trans = get_trans(img, atmo)
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)
    result = recover(img, trans_guided, atmo)

    return result.astype(np.float32)


if __name__ == '__main__':
    path = r"G:\Digital Image Processing\Test_img\0010_0.95_0.16.jpg"
    im = cv2.imread(path)
    res = DCP_dehaze(im)
    plt.title('DCP')
    plt.imshow(cv2.cvtColor(res.astype(np.float32), cv2.COLOR_BGR2RGB))
    plt.show()
