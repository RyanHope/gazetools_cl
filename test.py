#!/usr/bin/env python

import pkg_resources
import pyopencl as cl
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
cv2.waitKey(0)
cv2.destroyAllWindows()