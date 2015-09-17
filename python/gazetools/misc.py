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

class turnpoints_OCL(OCLWrapper):
    __kernel__ = "turnpoints.cl"
    def __call__(self, ctx, src, span=1, threshold=0):
        self.build(ctx)
        src = np.array(src, dtype=np.float32, copy=False)
        src_padded = np.zeros(src.shape[0]+2*span, dtype=src.dtype)
        src_padded[span:-span] = src
        src_padded[:span] = src_padded[span]
        src_padded[-span:] = src_padded[-span]
        src_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=src_padded)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, src.nbytes)
        queue = cl.CommandQueue(self.ctx)
        self.prg.turnpoints(queue, src.shape, None, src_buf, np.uint32(span), np.float32(threshold), out_buf)
        out = np.empty_like(src)
        cl.enqueue_read_buffer(queue, out_buf, out).wait()
        out = out.copy()
        src_buf.release()
        out_buf.release()
        return np.array(out, dtype=np.int32)
turnpoints = turnpoints_OCL()
