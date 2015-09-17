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

filters = [savgol_coeffs(33, 2, d, 1.0/500) for d in xrange(3)]

gaze = np.array(df[["smi_sxl","smi_syl","smi_sxl","smi_syl"]],dtype=np.float32)

smooth = convolve1d(ctx, gaze, filters[0])
vel = convolve1d(ctx, gaze, filters[1])

combo = convolve1d2(ctx, gaze, filters[1], filters[2])

t = np.arange(gaze.shape[0])

fig = plt.figure(figsize=(16,9))

a=fig.add_subplot(3,1,1)
plt.plot(t,gaze[:,0],'r-')
a.set_title("x")

a=fig.add_subplot(3,1,2)
plt.plot(t,combo[:,0],'b-')
a.set_title("v")
a=fig.add_subplot(3,1,3)
plt.plot(t,combo[:,2],'g-')
a.set_title("a")

plt.show()
