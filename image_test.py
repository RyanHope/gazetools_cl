#!/usr/bin/env python

import pyopencl as cl
import numpy as np
import cv2

def loadProgram(filename):
    with open(filename, 'r') as f:
        return "".join(f.readlines())

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
build_opts = "-I."
RGB2YCrCb = cl.Program(ctx, loadProgram("RGB2YCrCb.cl")).build(build_opts).RGB2YCrCb
YCrCb2RGB = cl.Program(ctx, loadProgram("YCrCb2RGB.cl")).build(build_opts).YCrCb2RGB

im = cv2.imread("PM5544_with_non-PAL_signals.png",cv2.IMREAD_UNCHANGED)
im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)

fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)

src = np.array(im)
src_buf = cl.image_from_array(ctx, src, 4)
dest_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(im.shape[1],im.shape[0]))

RGB2YCrCb(queue, (im.shape[1],im.shape[0]), None, src_buf, dest_buf)

dest = np.empty_like(src)
cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(im.shape[1],im.shape[0]))

Y = cv2.merge((dest[:,:,0],dest[:,:,0],dest[:,:,0]))
Cr = cv2.merge((dest[:,:,1],dest[:,:,1],dest[:,:,1]))
Cb = cv2.merge((dest[:,:,2],dest[:,:,2],dest[:,:,2]))

src2_buf = cl.image_from_array(ctx, dest, 4)
dest2_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(im.shape[1],im.shape[0]))

YCrCb2RGB(queue, (im.shape[1],im.shape[0]), None, src2_buf, dest2_buf)

dest2 = np.empty_like(dest)
cl.enqueue_copy(queue, dest2, dest2_buf, origin=(0, 0), region=(im.shape[1],im.shape[0]))

cv2.imshow('orig', cv2.cvtColor(src, cv2.COLOR_RGBA2BGR))
cv2.imshow('Y', Y)
cv2.imshow('Cr', Cr)
cv2.imshow('Cb', Cb)
cv2.imshow('CL_RGB2YCrCb -> CV_YCrCb2BGR', cv2.cvtColor(dest[:,:,0:3], cv2.COLOR_YCrCb2BGR))
cv2.imshow('CL_YCrCb2RGB(CL_RGB2YCrCb) -> CV_RGBA2BGR', cv2.cvtColor(dest2, cv2.COLOR_RGBA2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
