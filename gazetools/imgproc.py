from .helpers import OCLWrapper

import pyopencl as cl
import numpy as np
import os

class RGB2YCrCb_OCL(OCLWrapper):
    __kernel__ = "RGB2YCrCb.cl"
    def __call__(self, ctx, src2):
        self.build(ctx)
        src2 = np.asarray(src2)
        src = np.zeros((src2.shape[0], src2.shape[1], 4),dtype=src2.dtype)
        src[:,:,0:src2.shape[2]] = src2[:,:,0:src2.shape[2]]
        norm = np.issubdtype(src.dtype, np.integer)
        src_buf = cl.image_from_array(self.ctx, src, 4, norm_int=norm)
        dest_buf = cl.image_from_array(self.ctx, src, 4, mode="w", norm_int=norm)
        dest = np.empty_like(src)
        queue = cl.CommandQueue(self.ctx)
        self.prg.RGB2YCrCb(queue, (src.shape[1], src.shape[0]), None, src_buf, dest_buf)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(src.shape[1], src.shape[0])).wait()
        dest = dest[:,:,0:src2.shape[2]].copy()
        src_buf.release()
        dest_buf.release()
        return dest
RGB2YCrCb = RGB2YCrCb_OCL()

class YCrCb2RGB_OCL(OCLWrapper):
    __kernel__ = "YCrCb2RGB.cl"
    def __call__(self, ctx, src2):
        self.build(ctx)
        src2 = np.asarray(src2)
        src = np.zeros((src2.shape[0], src2.shape[1], 4),dtype=src2.dtype)
        src[:,:,0:src2.shape[2]] = src2[:,:,0:src2.shape[2]]
        norm = np.issubdtype(src.dtype, np.integer)
        src_buf = cl.image_from_array(self.ctx, src, 4, norm_int=norm)
        dest_buf = cl.image_from_array(self.ctx, src, 4, mode="w", norm_int=norm)
        dest = np.empty_like(src)
        queue = cl.CommandQueue(self.ctx)
        self.prg.YCrCb2RGB(queue, (src.shape[1], src.shape[0]), None, src_buf, dest_buf)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(src.shape[1], src.shape[0])).wait()
        dest = dest[:,:,0:src2.shape[2]].copy()
        src_buf.release()
        dest_buf.release()
        return dest
YCrCb2RGB = YCrCb2RGB_OCL()
