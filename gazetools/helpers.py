import pkg_resources
import pyopencl as cl

def getKernel(filename):
    return pkg_resources.resource_filename("cl",None), pkg_resources.resource_string("cl",filename)

class OCLWrapper(object):
    def __init__(self):
        self.ctx = None
    def build(self, ctx):
        if self.ctx != ctx:
            self.ctx = ctx
            inc, src = getKernel(self.__kernel__)
            self.prg = cl.Program(ctx, src).build("-I%s" % inc)
            print dir(self.prg)
            print self.prg.kernel_names
