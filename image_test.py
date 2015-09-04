#!/usr/bin/env python

import pyopencl as cl
import numpy as np
import cv2
from PIL import Image as PImage

def loadProgram(filename):
    with open(filename, 'r') as f:
        return "".join(f.readlines())

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
build_opts = "-I."
rgba2yuv = cl.Program(ctx, loadProgram("rgba2yuv.cl")).build(build_opts).rgba2yuv

im = PImage.open("PM5544_with_non-PAL_signals.png")
if im.mode != "RGBA":
    im = im.convert("RGBA")
fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)

src = np.array(im)
src_buf = cl.image_from_array(ctx, src, 4)
dest_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=im.size)

rgba2yuv(queue, im.size, None, src_buf, dest_buf)

dest = np.empty_like(src)
cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=im.size)

out = cv2.cvtColor(np.reshape(dest,(im.size[1],im.size[0],4)),cv2.COLOR_RGBA2BGR)
#out = cv2.merge((out[:,:,0],out[:,:,0],out[:,:,0]))
cv2.imshow('image', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
