#===============================================================================
# This file is part of gazetools.
# Copyright (C) 2015 Ryan M. Hope <rmh3093@gmail.com>
#
# gazetools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gazetools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gazetools.  If not, see <http://www.gnu.org/licenses/>.
#===============================================================================

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
        out = np.empty_like(x)
        cl.enqueue_read_buffer(queue, out_buf, out).wait()
        x_buf.release()
        y_buf.release()
        ez_buf.release()
        ex_buf.release()
        ey_buf.release()
        out_buf.release()
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
        self.prg.subtended_angle_naive(queue, x1.shape, None, x1_buf, y1_buf, x2_buf, y2_buf, rx, ry, sw, sh, ez_buf, ex_buf, ey_buf, out_buf)
        out = np.empty_like(x1)
        cl.enqueue_read_buffer(queue, out_buf, out).wait()
        x1_buf.release()
        y1_buf.release()
        x2_buf.release()
        y2_buf.release()
        ez_buf.release()
        ex_buf.release()
        ey_buf.release()
        out_buf.release()
        return out
subtended_angle = subtended_angle_OCL()

class subtended_angle2_OCL(OCLWrapper):
    __kernel__ = "subtended_angle.cl"
    def __call__(self, ctx, x1, y1, x2, y2, rx, ry, sw, sh, ez, ex, ey):
        self.build(ctx)
        x1 = np.array(x1, dtype=np.float32, copy=False)
        y1 = np.array(y1, dtype=np.float32, copy=False)
        x2 = np.float32(x2)
        y2 = np.float32(y2)
        ez = np.float32(ez)
        ex = np.float32(ex)
        ey = np.float32(ey)
        rx = np.float32(rx)
        ry = np.float32(ry)
        sw = np.float32(sw)
        sh = np.float32(sh)
        x1_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x1)
        y1_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y1)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, x1.nbytes)
        queue = cl.CommandQueue(self.ctx)
        self.prg.subtended_angle2_naive(queue, x1.shape, None, x1_buf, y1_buf, x2, y2, rx, ry, sw, sh, ez, ex, ey, out_buf)
        out = np.empty_like(x1)
        cl.enqueue_read_buffer(queue, out_buf, out).wait()
        x1_buf.release()
        y1_buf.release()
        out_buf.release()
        return out
subtended_angle2 = subtended_angle2_OCL()
