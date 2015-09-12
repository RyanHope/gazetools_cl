#!/usr/bin/env python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))

from PIL import Image
import matplotlib.image as mpimg

import numpy as np
import pyopencl as cl
import pkg_resources

from gazetools.imgproc import RGB2YCrCb, YCrCb2RGB

test_image = pkg_resources.resource_filename("gazetools", "resources/images/PM5544_with_non-PAL_signals.png")

# def test_RGB2YCrCb2RGB_uint8():
#     ctx = cl.create_some_context(0)
#     src1 = np.asarray(Image.open(test_image))
#     ycrcb = RGB2YCrCb(ctx, src1)
#     rgb = YCrCb2RGB(ctx, ycrcb)
#     assert np.array_equal(src1, rgb)
#
# def test_RGB2YCrCb2RGB_float32():
#     ctx = cl.create_some_context(0)
#     src2 = mpimg.imread(test_image)
#     ycrcb = RGB2YCrCb(ctx, src2)
#     rgb = YCrCb2RGB(ctx, ycrcb)
#     assert np.array_equal(src2, rgb)
