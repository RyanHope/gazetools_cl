#!/usr/bin/env python

import pkg_resources
import pyopencl as cl
import numpy as np
import pandas as pd
import cv2

from gazetools import *

ctx = cl.create_some_context()

src = cv2.cvtColor(cv2.imread(pkg_resources.resource_filename("images","PM5544_with_non-PAL_signals.png"),cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA)
dest = RGB2YCrCb(ctx, src)
Y = cv2.merge((dest[:,:,0],dest[:,:,0],dest[:,:,0]))
Cr = cv2.merge((dest[:,:,1],dest[:,:,1],dest[:,:,1]))
Cb = cv2.merge((dest[:,:,2],dest[:,:,2],dest[:,:,2]))
dest2 = YCrCb2RGB(ctx, dest)

cv2.imshow('orig', cv2.cvtColor(src, cv2.COLOR_RGBA2BGR))
cv2.imshow('Y', Y)
cv2.imshow('Cr', Cr)
cv2.imshow('Cb', Cb)
cv2.imshow('CL_RGB2YCrCb -> CV_YCrCb2BGR', cv2.cvtColor(dest, cv2.COLOR_YCrCb2BGR))
cv2.imshow('CL_YCrCb2RGB(CL_RGB2YCrCb) -> CV_RGBA2BGR', cv2.cvtColor(dest2, cv2.COLOR_RGB2BGR))
#cv2.waitKey(0)
cv2.destroyAllWindows()

print distance_2_point(ctx ,[0.0,0.0,1680.0/2.0,1680.0/2.0], [0.0,1050.0/2.0,0.0,1050/2.0], 1680,1050,473.76,296.1, [700.0]*4,[0.0]*4,[0.0]*4)
print distance_2_point(ctx, [1680/2], [1050/2], 1680,1050,473.76,296.1, [700]*1,[0]*1,[0]*1)
print subtended_angle(ctx, [0], [1050/2], [1680], [1050/2], 1680,1050,473.76,296.1, [700]*1,[0]*1,[0]*1)
N = 1680 * 1050
print subtended_angle(ctx, np.tile(np.arange(1680),1050), np.repeat(np.arange(1050), 1680), [1680/2]*N, [1050/2]*N, 1680,1050,473.76,296.1, [700]*N,[0]*N,[0]*N)
#
# smooth = savgol_coeffs(11, 2, 0, 1.0/500.0)
# df=pd.read_csv(pkg_resources.resource_filename("data","smi.csv"))
