#!/usr/bin/env python

import pkg_resources
import pyopencl as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.ndimage import convolve1d as convolve1d_np

from gazetools import *

ctx = cl.create_some_context()
kernel = [
    [1/16., 1/8., 1/16.],
    [1/8., 1/4., 1/8.],
    [1/16., 1/8., 1/16.],
]
src = cv2.cvtColor(cv2.imread(pkg_resources.resource_filename("images","PM5544_with_non-PAL_signals.png"),cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA)
dest1 = convolve2d(ctx, src, kernel)
dest2 = convolve2d(ctx, src, kernel)
print "==============================="
print dest1.astype('B')
print "- - - - - - - - - - - - - - - -"
print dest2.astype('B')
print "==============================="
print np.array_equal(dest1,dest2)

fig = plt.figure(figsize=(30,15))
a=fig.add_subplot(1,2,1)
plt.imshow(dest1.astype('B'))
a.set_title("blurred 1")
a=fig.add_subplot(1,2,2)
plt.imshow(dest2.astype('B'))
a.set_title("blurred 2")
plt.show()
