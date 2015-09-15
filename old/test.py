#!/usr/bin/env python

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../python"))

import pkg_resources
import pyopencl as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from gazetools import *

def mirror_padding(ctx, src, kernel):
    kernel = np.array(kernel, copy=False, dtype=np.float32)
    halflen = kernel.shape[0] / 2

    src = np.asarray(src)
    src_padded = np.full((src.shape[0]+2*halflen, src.shape[1]+2*halflen, 4), np.iinfo(src.dtype).max/2, dtype=src.dtype)
    src_padded[halflen:-halflen,halflen:-halflen,:src.shape[2]] = src[:,:,:src.shape[2]]

    src_padded[halflen:-halflen,:halflen,:src.shape[2]] = src_padded[halflen:-halflen,halflen:halflen*2,:src.shape[2]][:,::-1]
    src_padded[halflen:-halflen,-halflen:,:src.shape[2]] = src_padded[halflen:-halflen,-halflen*2:-halflen,:src.shape[2]][:,::-1]

    src_padded[:halflen,:,:src.shape[2]] = src_padded[halflen:halflen*2,:,:src.shape[2]][::-1,...]
    src_padded[-halflen:,:,:src.shape[2]] = src_padded[-halflen*2:-halflen,:,:src.shape[2]][::-1,...]

    dest = src_padded
    dest = dest[halflen:-halflen,halflen:-halflen,0:src.shape[2]].copy()
    return dest

    # src_padded = np.zeros(src.shape[0]+halflen*2, dtype=np.float32)
    # src_padded[halflen:-halflen] = src
    # src_padded[:halflen] = src[:halflen][::-1]
    # src_padded[src.shape[0]+halflen:] = src[src.shape[0]-halflen:][::-1]

ctx = cl.create_some_context(answers=[0,1])

orig = np.asarray(Image.open(pkg_resources.resource_filename("gazetools", "resources/images/PM5544_with_non-PAL_signals.png")))

#mp = mirror_padding(ctx, orig, kernel_gaussian5x5)

blur = convolve2d(ctx, orig, kernel_gaussian5x5)
print blur

plt.figure(1)
plt.imshow(orig)
plt.figure(2)
plt.imshow(blur)
plt.show()
