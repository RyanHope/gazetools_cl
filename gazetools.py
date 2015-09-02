#!/usr/bin/env python

import pyopencl as cl
import numpy as np

def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None):
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")
    halflen, rem = divmod(window_length, 2)
    if rem == 0:
        raise ValueError("window_length must be odd.")
    if pos is None:
        pos = halflen
    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than "
                         "window_length.")
    x = np.arange(-pos, window_length - pos, dtype=float)[::-1]
    order = np.arange(polyorder + 1).reshape(-1, 1)
    if order.size == 1:
        A = np.atleast_2d(x ** order[0, 0])
    else:
        A = x ** order
    y = np.zeros(polyorder + 1)
    y[deriv] = np.math.factorial(deriv) / (delta ** deriv)
    coeffs, _, _, _ = np.linalg.lstsq(A, y)
    return coeffs

def loadProgram(filename):
    with open(filename, 'r') as f:
        return "".join(f.readlines())

class gazetools:

    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        build_opts = "-I."
        self.cl_distance_2_point = cl.Program(self.ctx, loadProgram("distance_2_point.cl")).build(build_opts).distance_2_point
        self.cl_subtended_angle = cl.Program(self.ctx, loadProgram("subtended_angle.cl")).build(build_opts).subtended_angle
        self.cl_resmap = cl.Program(self.ctx, loadProgram("resmap.cl")).build(build_opts).resmap

    def distance_2_point(self, x, y, rx, ry, sw, sh, ez, ex, ey):
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
        self.cl_distance_2_point(self.queue, x.shape, None, x_buf, y_buf, rx, ry, sw, sh, ez_buf, ex_buf, ey_buf, out_buf)
        out = np.empty_like(x)
        cl.enqueue_read_buffer(self.queue, out_buf, out).wait()
        return out

    def subtended_angle(self, x1, y1, x2, y2, rx, ry, sw, sh, ez, ex, ey):
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
        self.cl_subtended_angle(self.queue, x1.shape, None, x1_buf, y1_buf, x2_buf, y2_buf, rx, ry, sw, sh, ez_buf, ex_buf, ey_buf, out_buf)
        out = np.empty_like(x1)
        cl.enqueue_read_buffer(self.queue, out_buf, out).wait()
        return out

    def resmap(self, ecc, ce):
        ecc = np.array(ecc, dtype=np.float32, copy=False)
        ce = np.array(ce, dtype=np.float32, copy=False)
        nl = np.uint32(ce.shape[0]-1)
        ecc_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=ecc)
        ce_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ce)
        self.cl_resmap(self.queue, ecc.shape, None, ecc_buf, ce_buf, nl)
        out = np.empty_like(ecc)
        cl.enqueue_read_buffer(self.queue, ecc_buf, out).wait()
        return out

if __name__ == "__main__":
    GT = gazetools()
    # print GT.distance_2_point([0.0,0.0,1680.0/2.0,1680.0/2.0],
    #                           [0.0,1050.0/2.0,0.0,1050/2.0],
    #                           1680,1050,473.76,296.1,
    #                           [700.0]*4,[0.0]*4,[0.0]*4)
    # print GT.distance_2_point([1680/2],
    #                           [1050/2],
    #                           1680,1050,473.76,296.1,
    #                           [700]*1,[0]*1,[0]*1)
    # print GT.subtended_angle([0],
    #                          [1050/2],
    #                          [1680],
    #                          [1050/2],
    #                          1680,1050,473.76,296.1,
    #                          [700]*1,[0]*1,[0]*1)
    N = 1680 * 1050
    print GT.subtended_angle(np.tile(np.arange(1680),1050),
                             np.repeat(np.arange(1050), 1680),
                             [1680/2],
                             [1050/2],
                             1680,1050,473.76,296.1,
                             [700]*N,[0]*N,[0]*N)
    # N = 10000
    # print GT.subtended_angle(np.random.exponential(50,N),
    #                          np.random.exponential(200,N),
    #                          (1680),
    #                          (1050/2),
    #                          1680,1050,473.76,296.1,
    #                          [700]*N,[0]*N,[0]*N)
    # print savgol_coeffs(11, 2, 0, 1.0/500.0)
