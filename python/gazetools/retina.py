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
from .visualangle import subtended_angle, subtended_angle2
from .filter import pyrUp, pyrDown

import pyopencl as cl
import numpy as np

class blend_OCL(OCLWrapper):
    __kernel__ = "retina.cl"
    def __call__(self, ctx, pyramid, blendmap, x, y):
        self.build(ctx)
        pyramid = np.asarray(pyramid)
        blendmap = np.asarray(blendmap)
        pyramid2 = np.zeros((pyramid.shape[0], pyramid.shape[1], pyramid.shape[2], 4),dtype=pyramid.dtype)
        pyramid2[:,:,:,0:pyramid.shape[3]] = pyramid[:,:,:,0:pyramid.shape[3]]
        norm = np.issubdtype(pyramid2.dtype, np.integer)
        pyramid_buf = cl.image_from_array(self.ctx, pyramid2, 4, mode="r", norm_int=norm)
        blendmap_buf = cl.image_from_array(self.ctx, blendmap, 4, mode="r")
        dest = np.zeros_like(pyramid2[0],dtype=pyramid.dtype)
        dest_buf = cl.image_from_array(self.ctx, dest, 4, mode="w", norm_int=norm)

        queue = cl.CommandQueue(self.ctx)
        self.prg.blend(queue, (dest.shape[1], dest.shape[0]), None,
                       pyramid_buf, blendmap_buf,
                       np.uint32(dest.shape[1]-x), np.uint32(dest.shape[0]-y),
                       dest_buf)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(dest.shape[1], dest.shape[0])).wait()

        dest = dest[:,:,0:pyramid.shape[3]].copy()
        dest_buf.release()
        pyramid_buf.release()
        blendmap_buf.release()
        return dest
blend = blend_OCL()

class blendmap_OCL(OCLWrapper):
    __kernel__ = "retina.cl"
    def __call__(self, ctx, rx, ry, sw, sh, ez, ex, ey, levels, halfres_eccentricity, contrast_sensitivity, decay_constant):
        self.build(ctx)

        w = 2*rx
        h = 2*ry

        assert levels==6
        var = np.array([0.849, 0.4245, 0.21225, 0.106125, 0.0530625, 0.02653125], dtype=np.float32)
        horizontal_degree = subtended_angle(ctx,[0],[ry],[w],[ry],rx,ry,sw,sh,[ez],[ex],[ey])[0]
        freq = 0.5/(horizontal_degree/w)

        critical_eccentricity = [0.0]
        for l in xrange(levels):
            ecc = halfres_eccentricity * ( (np.log(1/contrast_sensitivity)*(1<<l)/(decay_constant*freq))-1 )
            if ecc > 90.0: ecc = 90.0
            critical_eccentricity.append(ecc)
        critical_eccentricity.append(90.0)
        critical_eccentricity = np.array(critical_eccentricity, dtype=np.float32)
        print ("Critical Eccentricities",["%.2f" % ce for ce in critical_eccentricity])

        ce_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=critical_eccentricity)
        var_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=var)
        dest = np.zeros((h, w, 4), dtype=np.float32)
        dest_buf = cl.image_from_array(self.ctx, dest, 4, mode="w")

        queue = cl.CommandQueue(self.ctx)
        self.prg.blendmap(queue, (dest.shape[1], dest.shape[0]), None,
                          ce_buf, var_buf,
                          np.float32(rx), np.float32(ry),
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

        return dest
blendmap = blendmap_OCL()

class RetinaFilter(object):

    def __init__(self, ctx, rx, ry, sw, sh, ez, ex=0, ey=0):
        self.ctx = ctx
        self.rx = rx
        self.ry = ry
        self.sw = sw
        self.sh = sh
        self.ez = ez
        self.ex = ex
        self.ey = ey
        self.levels = 6
        self.decay_constant = 0.106
        self.halfres_eccentricity = 2.3
        self.contrast_sensitivity = 1.0/64.0

        self.blendmap = blendmap(self.ctx,
                                 self.rx, self.ry,
                                 self.sw, self.sh,
                                 self.ez, self.ex, self.ey,
                                 self.levels,
                                 self.halfres_eccentricity,
                                 self.contrast_sensitivity,
                                 self.decay_constant)
        self.pyramid = None

    def makePyramid(self, img):

        o = np.max(img.shape)
        fs = int(np.log(o)/np.log(2))
        fs = 1<<fs
        if fs < o: fs *= 2

        G = np.zeros((fs,fs,4), dtype=img.dtype)
        G[:img.shape[0],:img.shape[1],:img.shape[2]] = img

        gpA = [G]
        for i in xrange(self.levels):
            G = pyrDown(self.ctx, G)
            gpA.append(G)
        gpB = []
        for i in xrange(self.levels):
            G = gpA[i].copy()
            for _ in xrange(i):
                G = pyrUp(self.ctx, G)
            gpB.append(G)
        return [b[:img.shape[0],:img.shape[1],:img.shape[2]] for b in gpB]

    def blend(self, pyramid, x, y):
        return blend(self.ctx, pyramid, self.blendmap, x, y)
