#!/usr/bin/env python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))

from PIL import Image
import matplotlib.image as mpimg

import numpy as np
import pyopencl as cl
import pkg_resources

from gazetools.filter import convolve2d, kernel_gaussian3x3

test_image = pkg_resources.resource_filename("gazetools", "resources/images/PM5544_with_non-PAL_signals.png")

def test_convolve2d_both_equal_uint8():
    ctx = cl.create_some_context(0)
    src1 = np.asarray(Image.open(pkg_resources.resource_filename("gazetools", "resources/images/PM5544_with_non-PAL_signals.png")))
    dest1s1 = convolve2d(ctx, src1, kernel_gaussian3x3)
    dest2s1 = convolve2d(ctx, src1, kernel_gaussian3x3)
    assert np.array_equal(dest1s1, dest2s1)

def test_convolve2d_both_equal_float32():
    ctx = cl.create_some_context(0)
    src2 = mpimg.imread(pkg_resources.resource_filename("gazetools", "resources/images/PM5544_with_non-PAL_signals.png"))
    dest1s2 = convolve2d(ctx, src2, kernel_gaussian3x3)
    dest2s2 = convolve2d(ctx, src2, kernel_gaussian3x3)
    assert np.array_equal(dest1s2, dest2s2)
