#!/usr/bin/env python

import pkg_resources
import pyopencl as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from gazetools import *

ctx = cl.create_some_context()

sx = 1680
sy = 1050
sw = 473.76
sh = 296.1
ez = 700.0
x = np.tile(np.arange(2*sx),2*sy)
y = np.repeat(np.arange(2*sy),2*sx)
ecc = subtended_angle2(ctx, x, y, sx, sy, 2*sx, 2*sy, 2*sw, 2*sh, ez, 0, 0)
ecc_img = np.reshape(ecc, (2*sy,2*sx))
print ecc_img

fig = plt.figure(figsize=(15,15))
a=fig.add_subplot(1,1,1)
plt.imshow(ecc_img.astype('B'))
plt.show()
