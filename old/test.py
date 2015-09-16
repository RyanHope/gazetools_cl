import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../python"))

import pkg_resources
import pyopencl as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from gazetools import *

ctx = cl.create_some_context(answers=[0,1])

df=pd.read_csv(pkg_resources.resource_filename("gazetools","resources/data/smi.csv"))

smooth = savgol_coeffs(11, 2, 0, 1.0/500.0)

gaze = np.array(df[["smi_sxl","smi_syl"]],dtype=np.float32)
print gaze
print gaze.shape

gaze_smoothed = convolve1d(ctx, gaze, smooth)
print gaze_smoothed
print gaze_smoothed.shape

# x_cl1 = convolve1d(ctx, x, smooth)
#
t = np.arange(gaze_smoothed.shape[0])
plt.figure(1)
plt.plot(t,gaze[:,0],'r-',t,gaze_smoothed[:,0],'b-')
#plt.plot(t,gaze,'r-',t,gaze_smoothed,'b-')
plt.show()
