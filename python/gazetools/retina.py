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

from .visualangle import subtended_angle, subtended_angle2

import pyopencl as cl
import numpy as np

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
        self.critical_eccentricity = [0.0]

        w = 2*self.rx
        h = 2*self.ry

        self.horizontal_degree = subtended_angle(self.ctx,[0],[self.ry],[w],[self.ry],self.rx,self.ry,self.sw,self.sh,[self.ez],[self.ex],[self.ey])[0]
        freq = 0.5/(self.horizontal_degree/w)
        for l in xrange(self.levels):
            ecc = self.halfres_eccentricity * ( (np.log(1/self.contrast_sensitivity)*(1<<l)/(self.decay_constant*freq))-1 )
            if ecc > 90.0: ecc = 90.0
            self.critical_eccentricity.append(ecc)
        self.critical_eccentricity.append(90.0)
        print self.critical_eccentricity

        x = np.tile(np.arange(w),h)
        y = np.repeat(np.arange(h),w)
        self.ecc = subtended_angle2(self.ctx, x, y, self.rx, self.ry, w, h, 2*self.sw, 2*self.sh, self.ez, self.ex, self.ey).reshape((h,w))
