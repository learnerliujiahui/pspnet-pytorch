# this file convert the color map to ground truth

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path = '/home/liujiahui/Documents/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE'
save_path = '/home/liujiahui/Documents/gt'
colormap_list = os.listdir(path)


impervious_surfaces = [255, 255, 255]
building = [0, 0, 255]
low_vegetation = [0, 255, 255]
tree = [0, 255, 0]
car = [255, 255, 0]
background = [255, 0, 0]

for colormap in colormap_list:

    img_colormap = np.array(Image.open(os.path.join(path, colormap)))
    shape = img_colormap.shape

    gt = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if all(img_colormap[i, j] == impervious_surfaces):
                gt[i, j] = 0
            elif all(img_colormap[i, j] == building):
                gt[i, j] = 1
            elif all(img_colormap[i, j] == low_vegetation):
                gt[i, j] = 2
            elif all(img_colormap[i, j] == tree):
                gt[i, j] = 3
            elif all(img_colormap[i, j] == car):
                gt[i, j] = 4
            else:
                gt[i, j] = 5
    gt_img = Image.fromarray(np.uint8(gt))
    gt_img.save(os.path.join(save_path, colormap.split('.')[0] + '.png'))
    print('finish {}'.format(colormap))

