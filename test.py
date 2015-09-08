#!/usr/bin/env python

import pkg_resources
import pyopencl as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve1d as convolve1d_np

from gazetools import *

ctx = cl.create_some_context()

src = cv2.cvtColor(cv2.imread(pkg_resources.resource_filename("images","PM5544_with_non-PAL_signals.png"),cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA)
plt.imshow(src)
kernel = [
    1/16., 1/8., 1/16.,
    1/8., 1/4., 1/8.,
    1/16., 1/8., 1/16.,
]
dest = convolve2d(ctx, src, kernel)
plt.imshow(dest)
plt.show()
