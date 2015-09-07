from .helpers import OCLWrapper

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

class convolve1d_OCL(OCLWrapper):
    __kernel__ = "convolve1d.cl"
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    def __call__(self, ctx, src, kernel):
        self.build(ctx)
        kernel = np.array(kernel, copy=False, dtype=np.float32)
        halflen = kernel.shape[0] / 2
        src_padded = np.zeros(src.shape[0]+halflen*2, dtype=np.float32)
        src_padded[halflen:halflen+src.shape[0]] = src
        src_padded_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=src_padded)
        kernel_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel)
        dest_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, src_padded.nbytes)
        queue = cl.CommandQueue(self.ctx)
        self.prg.convolve1d_naive(queue, (src.shape[0],), None, src_padded_buf, dest_buf, kernel_buf, np.uint32(kernel.shape[0]), np.uint32(halflen))
        queue.finish()
        dest = np.empty_like(src_padded)
        cl.enqueue_read_buffer(queue, dest_buf, dest)
        return dest[halflen:halflen+src.shape[0]]
convolve1d = convolve1d_OCL()
