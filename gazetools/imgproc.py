from .helpers import loadProgram

import os
import pyopencl as cl
import numpy as np

class RGB2YCrCb_OCL(object):
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    def __init__(self):
        self.ctx = None
    def build(self, ctx):
        if self.ctx != ctx:
            self.ctx = ctx
            base = os.path.join(os.path.dirname(__file__),"../cl")
            self.prg = cl.Program(ctx, loadProgram(os.path.join(base,"RGB2YCrCb.cl"))).build("-I%s" % base).RGB2YCrCb
    def __call__(self, ctx, src):
        self.build(ctx)
        shape = (src.shape[1], src.shape[0])
        src_buf = cl.image_from_array(self.ctx, src, 4)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, self.fmt, shape=shape)
        queue = cl.CommandQueue(self.ctx)
        self.prg(queue, shape, None, src_buf, dest_buf)
        dest = np.empty_like(src)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=shape)
        return dest

RGB2YCrCb = RGB2YCrCb_OCL()
