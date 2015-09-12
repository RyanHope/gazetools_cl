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

class RGB2YCrCb_OCL(OCLWrapper):
    __kernel__ = "RGB2YCrCb.cl"
    def __call__(self, ctx, src):
        self.build(ctx)
        src = np.asarray(src)
        src2 = np.zeros((src.shape[0], src.shape[1], 4),dtype=src.dtype)
        src2[:,:,0:src.shape[2]] = src[:,:,0:src.shape[2]]
        norm = np.issubdtype(src2.dtype, np.integer)
        src2_buf = cl.image_from_array(self.ctx, src2, 4, norm_int=norm)
        dest_buf = cl.image_from_array(self.ctx, src2, 4, mode="w", norm_int=norm)
        dest = np.empty_like(src2)
        queue = cl.CommandQueue(self.ctx)
        self.prg.RGB2YCrCb(queue, (src2.shape[1], src2.shape[0]), None, src2_buf, dest_buf)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(src2.shape[1], src2.shape[0])).wait()
        dest = dest[:,:,0:src.shape[2]].copy()
        src2_buf.release()
        dest_buf.release()
        return dest
RGB2YCrCb = RGB2YCrCb_OCL()
"""
Converts a RGB image to the full range YCrCb colorspace.

:param ctx: an OpenCL context
:param src: a RGB image
:type ctx: pyopencl._cl.Context
:type src: numpy.ndarray
:returns: `src` converted to YCrCb colorspace
:rtype: numpy.ndarray

.. seealso:: `YCrCb2RGB`
"""


class YCrCb2RGB_OCL(OCLWrapper):
    __kernel__ = "YCrCb2RGB.cl"
    def __call__(self, ctx, src):
        self.build(ctx)
        src = np.asarray(src)
        src2 = np.zeros((src.shape[0], src.shape[1], 4),dtype=src.dtype)
        src2[:,:,0:src.shape[2]] = src[:,:,0:src.shape[2]]
        norm = np.issubdtype(src2.dtype, np.integer)
        src2_buf = cl.image_from_array(self.ctx, src2, 4, norm_int=norm)
        dest_buf = cl.image_from_array(self.ctx, src2, 4, mode="w", norm_int=norm)
        dest = np.empty_like(src2)
        queue = cl.CommandQueue(self.ctx)
        self.prg.YCrCb2RGB(queue, (src2.shape[1], src2.shape[0]), None, src2_buf, dest_buf)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(src2.shape[1], src2.shape[0])).wait()
        dest = dest[:,:,0:src.shape[2]].copy()
        src2_buf.release()
        dest_buf.release()
        return dest
YCrCb2RGB = YCrCb2RGB_OCL()
"""
Converts a full range YCrCb image to the RGB colorspace.

:param ctx: an OpenCL context
:param src: a YCrCb image
:type ctx: pyopencl._cl.Context
:type src: numpy.ndarray
:returns: `src` converted to RGB colorspace
:rtype: numpy.ndarray

.. seealso:: `RGB2YCrCb`
"""
