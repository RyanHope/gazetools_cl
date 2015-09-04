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
RGB2YCbCr = cl.Program(ctx, loadProgram("RGB2YCbCr.cl")).build(build_opts).RGB2YCbCr
YCbCr2RGB = cl.Program(ctx, loadProgram("YCbCr2RGB.cl")).build(build_opts).YCbCr2RGB

im = PImage.open("PM5544_with_non-PAL_signals.png")
if im.mode != "RGBA":
    im = im.convert("RGBA")
fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)

src = np.array(im)
src_buf = cl.image_from_array(ctx, src, 4)
dest_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=im.size)

RGB2YCbCr(queue, im.size, None, src_buf, dest_buf)

dest = np.empty_like(src)
cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=im.size)
out = np.reshape(dest,(im.size[1],im.size[0],4))

Y = cv2.merge((out[:,:,0],out[:,:,0],out[:,:,0]))
U = cv2.merge((out[:,:,1],out[:,:,1],out[:,:,1]))
V = cv2.merge((out[:,:,2],out[:,:,2],out[:,:,2]))

src2_buf = cl.image_from_array(ctx, out, 4)
dest2_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=im.size)

YCbCr2RGB(queue, im.size, None, src2_buf, dest2_buf)

dest2 = np.empty_like(out)
cl.enqueue_copy(queue, dest2, dest2_buf, origin=(0, 0), region=im.size)
out2 = np.reshape(dest2,(im.size[1],im.size[0],4))

cv2.imshow('rgb', src)
# cv2.imshow('Y', Y)
# cv2.imshow('U', U)
# cv2.imshow('V', V)
cv2.imshow('YUVtoRGB', cv2.cvtColor(out[:,:,0:3], cv2.COLOR_YCrCb2BGR))
print out2
cv2.imshow('fromYUV', cv2.cvtColor(out2, cv2.COLOR_RGBA2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
