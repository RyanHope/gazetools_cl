from .helpers import OCLWrapper

import pyopencl as cl
import numpy as np
import os

class distance_2_point_OCL(OCLWrapper):
    __kernel__ = "distance_2_point.cl"
    def __call__(self, ctx, x, y, rx, ry, sw, sh, ez, ex, ey):
        self.build(ctx)
        x = np.array(x, dtype=np.float32, copy=False)
        y = np.array(y, dtype=np.float32, copy=False)
        ez = np.array(ez, dtype=np.float32, copy=False)
        ex = np.array(ex, dtype=np.float32, copy=False)
        ey = np.array(ey, dtype=np.float32, copy=False)
        rx = np.float32(rx)
        ry = np.float32(ry)
        sw = np.float32(sw)
        sh = np.float32(sh)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        y_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y)
        ez_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ez)
        ex_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ex)
        ey_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ey)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, x.nbytes)
        queue = cl.CommandQueue(self.ctx)
        self.prg.distance_2_point(queue, x.shape, None, x_buf, y_buf, rx, ry, sw, sh, ez_buf, ex_buf, ey_buf, out_buf)
        queue.finish()
        out = np.empty_like(x)
        cl.enqueue_read_buffer(queue, out_buf, out)
        return out
distance_2_point = distance_2_point_OCL()

class subtended_angle_OCL(OCLWrapper):
    __kernel__ = "subtended_angle.cl"
    def __call__(self, ctx, x1, y1, x2, y2, rx, ry, sw, sh, ez, ex, ey):
        self.build(ctx)
        x1 = np.array(x1, dtype=np.float32, copy=False)
        y1 = np.array(y1, dtype=np.float32, copy=False)
        x2 = np.array(x2, dtype=np.float32, copy=False)
        y2 = np.array(y2, dtype=np.float32, copy=False)
        ez = np.array(ez, dtype=np.float32, copy=False)
        ex = np.array(ex, dtype=np.float32, copy=False)
        ey = np.array(ey, dtype=np.float32, copy=False)
        rx = np.float32(rx)
        ry = np.float32(ry)
        sw = np.float32(sw)
        sh = np.float32(sh)
        x1_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x1)
        y1_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y1)
        x2_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x2)
        y2_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y2)
        ez_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ez)
        ex_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ex)
        ey_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ey)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, x1.nbytes)
        queue = cl.CommandQueue(self.ctx)
        self.prg.subtended_angle(queue, x1.shape, None, x1_buf, y1_buf, x2_buf, y2_buf, rx, ry, sw, sh, ez_buf, ex_buf, ey_buf, out_buf)
        queue.finish()
        out = np.empty_like(x1)
        cl.enqueue_read_buffer(queue, out_buf, out)
        return out
subtended_angle = subtended_angle_OCL()
