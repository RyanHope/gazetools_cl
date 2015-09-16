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

kernel_gaussian3x3 = np.array([
    [1/16., 2/16., 1/16.],
    [2/16., 4/16., 2/8.],
    [1/16., 2/16., 1/16.],
], dtype=np.float32)

kernel_gaussian5x5 = np.array([
    [1/256., 4/256., 6/256., 4/256., 1/256.],
    [4/256., 16/256., 24/256., 16/256., 4/256.],
    [6/256., 24/256., 36/256., 24/256., 6/256.],
    [4/256., 16/256., 24/256., 16/256., 4/256.],
    [1/256., 4/256., 6/256., 4/256., 1/256.],
], dtype=np.float32)

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
    return np.array(coeffs,dtype=np.float32)

class convolve1d_OCL(OCLWrapper):
    __kernel__ = "convolve1d.cl"
    def __call__(self, ctx, src, kernel):
        self.build(ctx)
        kernel = np.array(kernel, copy=False, dtype=np.float32)
        halflen = kernel.shape[0] / 2
        kernel_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel)

        inshape = src.shape
        src = np.asarray(src)
        dims = len(src.shape)
        if dims == 1:
            src = src.reshape(src.shape[0],1,1)
        elif dims == 2:
            src = src.reshape(src.shape[0],1,src.shape[1])

        src_padded = np.zeros((src.shape[0]+2*halflen, 1, 4), dtype=src.dtype)
        src_padded[halflen:-halflen,:,:src.shape[2]] = src[:,:,:src.shape[2]]

        src_padded[:halflen,:,:] = src_padded[halflen:halflen*2,:,:][::-1,...]
        src_padded[-halflen:,:,:] = src_padded[-halflen*2:-halflen,:,:][::-1,...]

        norm = np.issubdtype(src.dtype, np.integer)
        src_buf = cl.image_from_array(self.ctx, src_padded, 4, norm_int=norm)
        dest = np.zeros((src.shape[0], src.shape[1], 4), dtype=src.dtype)
        dest_buf = cl.image_from_array(self.ctx, dest, 4, mode="w", norm_int=norm)

        queue = cl.CommandQueue(self.ctx)
        self.prg.convolve1d_naive(queue, (dest.shape[1], dest.shape[0]), None, src_buf, dest_buf, kernel_buf, np.int32(halflen))
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(src.shape[1], src.shape[0])).wait()

        dest = dest[:,:,:src.shape[2]].reshape(inshape)

        src_buf.release()
        dest_buf.release()
        kernel_buf.release()
        return dest
convolve1d = convolve1d_OCL()

class convolve2d_OCL(OCLWrapper):
    __kernel__ = "convolve2d.cl"
    def __call__(self, ctx, src, kernel):
        self.build(ctx)
        kernel = np.array(kernel, copy=False, dtype=np.float32)
        halflen = kernel.shape[0] / 2
        kernelf = kernel.flatten()
        kernelf_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernelf)

        src = np.asarray(src)
        src_padded = np.zeros((src.shape[0]+2*halflen, src.shape[1]+2*halflen, 4), dtype=src.dtype)
        src_padded[halflen:-halflen,halflen:-halflen,:src.shape[2]] = src[:,:,:src.shape[2]]

        src_padded[halflen:-halflen,:halflen,:src.shape[2]] = src_padded[halflen:-halflen,halflen:halflen*2,:src.shape[2]][:,::-1]
        src_padded[halflen:-halflen,-halflen:,:src.shape[2]] = src_padded[halflen:-halflen,-halflen*2:-halflen,:src.shape[2]][:,::-1]

        src_padded[:halflen,:,:src.shape[2]] = src_padded[halflen:halflen*2,:,:src.shape[2]][::-1,...]
        src_padded[-halflen:,:,:src.shape[2]] = src_padded[-halflen*2:-halflen,:,:src.shape[2]][::-1,...]

        norm = np.issubdtype(src.dtype, np.integer)
        src_buf = cl.image_from_array(self.ctx, src_padded, 4, norm_int=norm)
        dest = np.zeros((src.shape[0], src.shape[1], 4), dtype=src.dtype)
        dest_buf = cl.image_from_array(self.ctx, dest, 4, mode="w", norm_int=norm)

        queue = cl.CommandQueue(self.ctx)
        self.prg.convolve2d_naive(queue, (dest.shape[1], dest.shape[0]), None, src_buf, dest_buf, kernelf_buf, np.int32(kernel.shape[0]))
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(src.shape[1], src.shape[0])).wait()

        dest = dest[:,:,:src.shape[2]].copy()

        src_buf.release()
        dest_buf.release()
        kernelf_buf.release()
        return dest
convolve2d = convolve2d_OCL()

"""
Blurs an image and downsamples it.

:param ctx: an OpenCL context
:param src: an image
:type ctx: pyopencl._cl.Context
:type src: numpy.ndarray
:returns: `src` downsampled to 1/4 of it's orignal area
:rtype: numpy.ndarray

.. seealso:: `pyrUp`
"""
def pyrDown(ctx, src):
    down = convolve2d(ctx, src, kernel_gaussian5x5)
    return down[1::2,1::2,:]

"""
Upsamples an image and then blurs it.

:param ctx: an OpenCL context
:param src: an image
:type ctx: pyopencl._cl.Context
:type src: numpy.ndarray
:returns: `src` upsampled to 4 times it's original area
:rtype: numpy.ndarray

.. seealso:: `pyrDown`
"""
def pyrUp(ctx, src):
    up = np.zeros((src.shape[0]*2,src.shape[1]*2,src.shape[2]),dtype=src.dtype)
    up[1::2,1::2,:] = src
    return convolve2d(ctx, up, 4*kernel_gaussian5x5)
