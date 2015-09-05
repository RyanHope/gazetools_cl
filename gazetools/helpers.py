import pkg_resources

def getKernel(filename):
    return pkg_resources.resource_filename("cl",None), pkg_resources.resource_string("cl",filename)
