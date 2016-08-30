#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/14/2016 2:28 PM
# @Author  : ANG LI
# @Affiliation    : University of Arkansas
# @File    : ImageFeature.py
# @Software: PyCharm

import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.exposure import equalize_hist

def show_corners(corners, image):
    fig=plt.figure()
    plt.gray()
    plt.imshow(image)
    y_corner, x_corner=zip(*corners)
    plt.plot(x_corner,y_corner,'or')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0],0)
    fig.set_size_inches(np.array(fig.get_size_inches())*1.5)
    plt.show()

