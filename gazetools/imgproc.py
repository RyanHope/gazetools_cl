from .helpers import getKernel

import pyopencl as cl
import numpy as np
import os

class RGB2YCrCb_OCL(object):
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    def __init__(self):
        self.ctx = None
    def build(self, ctx):
        if self.ctx != ctx:
            self.ctx = ctx
            inc, src = getKernel("RGB2YCrCb.cl")
            self.prg = cl.Program(ctx, src).build("-I%s" % inc).RGB2YCrCb
    def __call__(self, ctx, src):
        self.build(ctx)
        shape = (src.shape[1], src.shape[0])
        src_buf = cl.image_from_array(self.ctx, src, 4)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, self.fmt, shape=shape)
        queue = cl.CommandQueue(self.ctx)
        self.prg(queue, shape, None, src_buf, dest_buf)
        dest = np.empty_like(src)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=shape)
        return np.delete(dest, 3, 2)
RGB2YCrCb = RGB2YCrCb_OCL()

class YCrCb2RGB_OCL(object):
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    def __init__(self):
        self.ctx = None
    def build(self, ctx):
        if self.ctx != ctx:
            self.ctx = ctx
            inc, src = getKernel("YCrCb2RGB.cl")
            self.prg = cl.Program(ctx, src).build("-I%s" % inc).YCrCb2RGB
    def __call__(self, ctx, src):
        self.build(ctx)
        shape = (src.shape[1], src.shape[0])
        src2 = np.zeros((src.shape[0],src.shape[1],src.shape[2]+1),dtype=np.uint8)
        src2[:,:,0:3] = src
        src2_buf = cl.image_from_array(self.ctx, src2, 4)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, self.fmt, shape=shape)
        queue = cl.CommandQueue(self.ctx)
        self.prg(queue, shape, None, src2_buf, dest_buf)
        dest = np.empty_like(src2)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=shape)
        return np.delete(dest, 3, 2)
YCrCb2RGB = YCrCb2RGB_OCL()
