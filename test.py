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

smooth = savgol_coeffs(11, 2, 0, 1.0/500.0)
print smooth
df=pd.read_csv(pkg_resources.resource_filename("data","smi.csv"))
# print df.info()

x = np.array(df["smi_sxl"], dtype=np.float32)
x_np = convolve1d_np(x, smooth)
x_cl = convolve1d(ctx, x, smooth)
x_cl2 = convolve1d(ctx, x, smooth)
x_cl3 = convolve1d(ctx, x, smooth)

print x[11:22]
print x_np[11:22]
print x_cl[11:22]
print x_cl2[11:22]
print x_cl3[11:22]

t = np.arange(x.shape[0])
plt.plot(t,x_cl,'g-',t,x_cl2,'b-',t,x_cl3,'r-')
plt.show()
