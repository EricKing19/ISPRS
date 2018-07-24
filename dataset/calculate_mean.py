import os
import cv2
import os.path as osp
import numpy as np

img_dir = '/home/jinqizhao/dataset/image/Remote_sensing/potsdam/2_Ortho_RGB_seg/'
list_path = './list/top_potsdam.txt'


img_ids = [i_id.strip() for i_id in open(list_path)]
image_size = 0
sum_b = 0
sum_g = 0
sum_r = 0
count = 0

for name in img_ids:
    img_file = osp.join(img_dir, "%s.tif" % name.replace('label', 'RGB'))
    img = cv2.imread(img_file)
    sum_b += img[:,:,0].mean()
    sum_g += img[:,:,1].mean()
    sum_r += img[:,:,2].mean()
    count += 1

sum_b /= count
sum_g /= count
sum_r /= count
img_mean = [sum_b, sum_g, sum_r]
print(img_mean)
print(count)
