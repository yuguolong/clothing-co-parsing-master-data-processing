from glob import glob
import os
from scipy.io import loadmat
import cv2
import numpy as np
import time
from numba import jit
from numba import autojit

img_list = glob('./label/'+'*.jpg')

for i in img_list:
    img = cv2.imread(i)
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=7)
    img = cv2.erode(img, kernel, iterations=2)
    name = i.split('\\')[-1]
    # print(name)

    cv2.imwrite('./label1/{}'.format(name),img)