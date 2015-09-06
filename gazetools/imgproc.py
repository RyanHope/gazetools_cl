from .helpers import OCLWrapper

import pyopencl as cl
import numpy as np
import os

class RGB2YCrCb_OCL(OCLWrapper):
    __kernel__ = "RGB2YCrCb.cl"
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    def __call__(self, ctx, src):
        self.build(ctx)
        shape = (src.shape[1], src.shape[0])
        src_buf = cl.image_from_array(self.ctx, src, 4)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, self.fmt, shape=shape)
        queue = cl.CommandQueue(self.ctx)
        self.prg.RGB2YCrCb(queue, shape, None, src_buf, dest_buf)
        queue.finish()
        dest = np.empty_like(src)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=shape)
        return np.delete(dest, 3, 2)
RGB2YCrCb = RGB2YCrCb_OCL()

class YCrCb2RGB_OCL(OCLWrapper):
    __kernel__ = "YCrCb2RGB.cl"
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    def __call__(self, ctx, src):
        self.build(ctx)
        shape = (src.shape[1], src.shape[0])
        src2 = np.zeros((src.shape[0],src.shape[1],src.shape[2]+1),dtype=np.uint8)
        src2[:,:,0:3] = src
        src2_buf = cl.image_from_array(self.ctx, src2, 4)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, self.fmt, shape=shape)
        queue = cl.CommandQueue(self.ctx)
        self.prg.YCrCb2RGB(queue, shape, None, src2_buf, dest_buf)
        queue.finish()
        dest = np.empty_like(src2)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=shape)
        return np.delete(dest, 3, 2)
YCrCb2RGB = YCrCb2RGB_OCL()
