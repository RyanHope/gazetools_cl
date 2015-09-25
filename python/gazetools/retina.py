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

from .helpers import OCLWrapper, init_image
from .visualangle import subtended_angle, subtended_angle2
from .filter import pyrUp, pyrDown

import pyopencl as cl
import numpy as np
import cv2

class blend_OCL(OCLWrapper):
    __kernel__ = "retina.cl"
    def __call__(self, ctx, pyramid, blendmap, x, y):
        self.build(ctx)

        norm = np.issubdtype(pyramid.dtype, np.integer)

        pyramid_buf = cl.image_from_array(self.ctx, pyramid, 4, mode="r", norm_int=norm)

        dest = np.zeros_like(pyramid[:,:,0,:],dtype=pyramid.dtype)
        dest_buf = init_image(self.ctx, dest, 4, mode="w", norm_int=norm)

        xoff = dest.shape[1] - x
        yoff = dest.shape[0] - y
        blendmap_buf = cl.image_from_array(self.ctx, blendmap[yoff:yoff+dest.shape[0],xoff:xoff+dest.shape[1],:].copy(), 4, mode="r")

        queue = cl.CommandQueue(self.ctx)
        self.prg.blend(queue, (dest.shape[1], dest.shape[0]), None,
                       pyramid_buf, blendmap_buf, dest_buf)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(dest.shape[1], dest.shape[0])).wait()

        dest_buf.release()
        pyramid_buf.release()
        blendmap_buf.release()
        return dest
blend = blend_OCL()

class blendmap_OCL(OCLWrapper):
    __kernel__ = "retina.cl"
    def __call__(self, ctx, ix, iy, rx, ry, sw, sh, ez, ex, ey, levels, halfres_eccentricity, contrast_sensitivity, decay_constant):
        self.build(ctx)

        w = 2*ix
        h = 2*iy

        assert levels==6
        var = np.array([0.849, 0.4245, 0.21225, 0.106125, 0.0530625, 0.02653125], dtype=np.float32)
        horizontal_degree = subtended_angle(ctx,[0],[ry],[2*rx],[ry],rx,ry,sw,sh,[ez],[ex],[ey])[0]
        freq = 0.5/(horizontal_degree/(2*rx))

        critical_eccentricity = [0.0]
        for l in xrange(levels):
            ecc = halfres_eccentricity * ( (np.log(1/contrast_sensitivity)*(1<<l)/(decay_constant*freq))-1 )
            if ecc > 90.0: ecc = 90.0
            critical_eccentricity.append(ecc)
        critical_eccentricity.append(90.0)
        critical_eccentricity = np.array(critical_eccentricity, dtype=np.float32)

        ce_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=critical_eccentricity)
        var_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=var)
        dest = np.zeros((h, w, 4), dtype=np.float32)
        dest_buf = cl.image_from_array(self.ctx, dest, 4, mode="w")

        queue = cl.CommandQueue(self.ctx)
        self.prg.blendmap(queue, (dest.shape[1], dest.shape[0]), None,
                          ce_buf, var_buf,
                          np.float32(ix), np.float32(iy),
                          np.float32(w), np.float32(h),
                          np.float32(2*sw), np.float32(2*sh),
                          np.float32(ez), np.float32(ex), np.float32(ey),
                          np.uint32(levels),
                          dest_buf)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(dest.shape[1], dest.shape[0])).wait()

        dest = dest.copy()
        dest_buf.release()
        ce_buf.release()
        var_buf.release()

        return critical_eccentricity, dest
blendmap = blendmap_OCL()

class RetinaFilter(object):

    #rf = RetinaFilter(ctx,img.shape[1],img.shape[0],vs_rx,vs_sw,vs_pd,vs_aspect,ez)
    def __init__(self, ctx, ix, iy, idtype, rx, sw, pd, ez, ex=0, ey=0):
        self.ctx = ctx
        self.ix = ix
        self.iy = iy
        self.rx = rx
        asp = 1.0 * iy / ix
        self.ry = 1.0 * rx * asp
        self.sw = sw
        self.sh = 1.0 * sw * asp
        self.ez = ez
        self.ex = ex
        self.ey = ey
        self.levels = 6
        self.decay_constant = 0.106
        self.halfres_eccentricity = 2.3
        self.contrast_sensitivity = 1.0/64.0

        self.critical_eccentricity, self.blendmap = blendmap(self.ctx,
                                 self.ix, self.iy,
                                 self.rx, self.ry,
                                 self.sw, self.sh,
                                 self.ez, self.ex, self.ey,
                                 self.levels,
                                 self.halfres_eccentricity,
                                 self.contrast_sensitivity,
                                 self.decay_constant)
        o = np.max((self.ix,self.iy))
        self.fs = int(np.log(o)/np.log(2))
        self.fs = 1<<self.fs
        if self.fs < o: self.fs *= 2
        self.pyramid = np.zeros((self.fs,self.fs,self.levels,4), dtype=idtype)

    def filter(self, img, x, y):
        self.pyramid[:img.shape[0],:img.shape[1],0,:img.shape[2]] = img
        for i in xrange(self.levels-1):
            self.pyramid[:self.fs>>(i+1),:self.fs>>(i+1),i+1,:] = cv2.pyrDown(self.pyramid[:self.fs>>(i),:self.fs>>(i),i,:])
        for i in xrange(self.levels):
            for j in xrange(i):
                self.pyramid[:self.fs>>(i-j-1),:self.fs>>(i-j-1),i,:] = cv2.pyrUp(self.pyramid[:self.fs>>(i-j),:self.fs>>(i-j),i,:])
        return blend(self.ctx, self.pyramid[:img.shape[0],:img.shape[1],:,:].copy(), self.blendmap, x, y)
