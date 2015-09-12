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

kernel_gaussian3x3 = [
    [1/16., 1/8., 1/16.],
    [1/8., 1/4., 1/8.],
    [1/16., 1/8., 1/16.],
]

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
    x = np.arange(-pos, window_length - pos, dtype=np.float32)[::-1]
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
    def __call__(self, ctx, src, kernel):
        self.build(ctx)
        src = np.array(src, copy=False, dtype=np.float32)
        kernel = np.array(kernel, copy=False, dtype=np.float32)
        halflen = kernel.shape[0] / 2
        src_padded = np.zeros(src.shape[0]+halflen*2, dtype=np.float32)
        src_padded[halflen:-halflen] = src
        src_padded[:halflen] = src[:halflen][::-1]
        src_padded[src.shape[0]+halflen:] = src[src.shape[0]-halflen:][::-1]
        src_padded_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=src_padded)
        kernel_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel)
        dest_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, src.nbytes)
        queue = cl.CommandQueue(self.ctx)
        self.prg.convolve1d_naive(queue, src.shape, None, src_padded_buf, dest_buf, kernel_buf, np.uint32(kernel.shape[0]), np.uint32(halflen))
        dest = np.empty_like(src, dtype=np.float32)
        cl.enqueue_read_buffer(queue, dest_buf, dest).wait()
        src_padded_buf.release()
        kernel_buf.release()
        dest_buf.release()
        return dest
convolve1d = convolve1d_OCL()

class convolve2d_OCL(OCLWrapper):
    __kernel__ = "convolve2d.cl"
    def __call__(self, ctx, src2, kernel):
        self.build(ctx)
        src2 = np.asarray(src2)
        src = np.zeros((src2.shape[0], src2.shape[1], 4),dtype=src2.dtype)
        src[:,:,0:src2.shape[2]] = src2[:,:,0:src2.shape[2]]
        norm = np.issubdtype(src.dtype, np.integer)
        src_buf = cl.image_from_array(self.ctx, src, 4, norm_int=norm)
        dest_buf = cl.image_from_array(self.ctx, src, 4, mode="w", norm_int=norm)
        dest = np.empty_like(src)
        kernel = np.array(kernel, dtype=np.float32)
        kernelf = kernel.flatten()
        kernelf_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernelf)
        halflen = (kernelf.shape[0]>>1)
        queue = cl.CommandQueue(self.ctx)
        self.prg.convolve2d_naive(queue, (src.shape[1]-halflen, src.shape[0]-halflen), None, src_buf, dest_buf, kernelf_buf, np.int_(kernelf.shape[0]))
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(src.shape[1], src.shape[0])).wait()
        dest = dest[:,:,0:src2.shape[2]].copy()
        src_buf.release()
        dest_buf.release()
        kernelf_buf.release()
        return dest
convolve2d = convolve2d_OCL()
