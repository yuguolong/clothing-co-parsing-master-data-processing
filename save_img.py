from glob import glob
import os
from scipy.io import loadmat
import cv2
import numpy as np
import time
from numba import jit
from numba import autojit

img_list = glob('./annotations/pixel-level/'+'*.mat')
isExists = os.path.exists('./label')

if not isExists:
    os.makedirs('./label')

start_time = time.time()

@jit(nopython=True)
def save(img,img1,x,y):
    for i in range(x):
        for j in range(y):
            if img[i][j] != 0:
                img1[i][j] = 255
            # else:
            #     img1[i][j] = 255

    return img1

count = 0
for imgs in img_list:
    m = loadmat('{}'.format(imgs))  # 读出来的 m 是一个dict（字典）数据结构
    img = m['groundtruth']

    img = np.array(img)
    img1 = img
    x,y = img.shape
    list = [0,41]
    img1 = save(img,img1,x,y)


    img_name = imgs.split('\\')[-1].split('.')[0]
    cv2.imwrite('./label/{}.jpg'.format(img_name), img1)
    count+=1
    # if count>20:
    #     break

end_time = time.time()
print(end_time-start_time)

#44.91693377494812

#避免重复调用
#45.992815017700195

#优化一
#46.89664077758789

#jit
#46.91658806800842

#优化三
#0.45179247856140137
#很牛

#传了列表进去
#16.54876399040222

#不传列表
#8.372618675231934