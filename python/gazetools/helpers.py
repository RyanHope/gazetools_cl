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

import pkg_resources, os
import pyopencl as cl

def getKernel(filename):
    return pkg_resources.resource_filename(__name__,"resources/cl"), pkg_resources.resource_string(__name__,os.path.join('resources/cl',filename))

class OCLWrapper(object):
    def __init__(self):
        self.ctx = None
    def build(self, ctx):
        if self.ctx != ctx:
            self.ctx = ctx
            inc, src = getKernel(self.__kernel__)
            self.prg = cl.Program(self.ctx, src).build("-I%s" % inc)
