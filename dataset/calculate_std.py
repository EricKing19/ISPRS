import os
import cv2
import os.path as osp
import math

img_dir = '/home/jinqizhao/dataset/image/Remote_sensing/potsdam/2_Ortho_RGB_seg/'
list_path = '/home/jinqizhao/pycharm-edu-4.0.2/PyCharmCode/ISPRS/dataset/list/top_potsdam.txt'

img_mean = [84.9966433081362, 91.78542590582812, 85.861383254881261]
img_ids = [i_id.strip() for i_id in open(list_path)]
var_b = 0
var_g = 0
var_r = 0
count = 0

for name in img_ids:
    img_file = osp.join(img_dir, "%s.tif" % name.replace('label', 'RGB'))
    img = cv2.imread(img_file)
    # img : BGR
    img = img.astype('int32')
    var_b += ((img[:,:,0] - img_mean[0])*(img[:,:,0] - img_mean[0])).mean()
    var_g += ((img[:,:,1] - img_mean[1])*(img[:,:,1] - img_mean[1])).mean()
    var_r += ((img[:,:,2] - img_mean[2])*(img[:,:,2] - img_mean[2])).mean()
    count += 1

var_b /= count
var_g /= count
var_r /= count
var_b = math.sqrt(var_b)
var_g = math.sqrt(var_g)
var_r = math.sqrt(var_r)
img_var = [var_b, var_g, var_r]
print(img_var)
print(count)
