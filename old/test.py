#!/usr/bin/env python

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

#orig = np.asarray(Image.open(pkg_resources.resource_filename("gazetools", "resources/images/PM5544_with_non-PAL_signals.png")))
orig = mpimg.imread(pkg_resources.resource_filename("gazetools", "resources/images/PM5544_with_non-PAL_signals.png"))

down1 = pyrDown(ctx, orig)
down2 = pyrDown(ctx, down1)
down3 = pyrDown(ctx, down2)
down4 = pyrDown(ctx, down3)
down5 = pyrDown(ctx, down4)

combined = np.zeros((orig.shape[0]+orig.shape[0]/2,orig.shape[1],orig.shape[2]),dtype=orig.dtype)
combined[:orig.shape[0],:orig.shape[1],:] = orig
combined[orig.shape[0]:orig.shape[0]+down1.shape[0],:down1.shape[1],:] = down1
combined[orig.shape[0]+down2.shape[0]:orig.shape[0]+2*down2.shape[0],down1.shape[1]:down1.shape[1]+down2.shape[1]] = down2
combined[orig.shape[0]+down2.shape[0]+down3.shape[0]:orig.shape[0]+2*down2.shape[0]+2*down3.shape[0],down1.shape[1]+down2.shape[1]:down1.shape[1]+down2.shape[1]+down3.shape[1]] = down3
combined[orig.shape[0]+down2.shape[0]+down3.shape[0]+down4.shape[0]:orig.shape[0]+2*down2.shape[0]+2*down3.shape[0]+2*down4.shape[0],down1.shape[1]+down2.shape[1]+down3.shape[1]:down1.shape[1]+down2.shape[1]+down3.shape[1]+down4.shape[1]] = down4
combined[orig.shape[0]+down2.shape[0]+down3.shape[0]+down4.shape[0]+down5.shape[0]:orig.shape[0]+2*down2.shape[0]+2*down3.shape[0]+2*down4.shape[0]+2*down5.shape[0],down1.shape[1]+down2.shape[1]+down3.shape[1]+down4.shape[1]:down1.shape[1]+down2.shape[1]+down3.shape[1]+down4.shape[1]+down5.shape[1]] = down5

plt.imshow(combined)
plt.show()
