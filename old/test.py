import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../python"))

import pkg_resources
import pyopencl as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mc
from PIL import Image

from gazetools import *

ctx = cl.create_some_context(answers=[0,1])

# img = np.asarray(Image.open(pkg_resources.resource_filename("gazetools", "resources/images/Bl9ZQQG.jpg")))
img = np.asarray(Image.open(pkg_resources.resource_filename("gazetools", "resources/images/PM5544_with_non-PAL_signals.png")))
r = 1.0*img.shape[1]/img.shape[0]
rr = 10
aspect = (r*rr,rr)
# plt.figure("Original",figsize=aspect)
# plt.imshow(img)
print ("Source Image",img.dtype,img.shape)

vs_pd = 3.546
vs_sw = 473.76
vs_rx = 1680
ez = 700
rf = RetinaFilter(ctx,img.shape[1],img.shape[0],vs_rx,vs_sw,vs_pd,ez)
print ("Critical Eccentricities",["%.2f" % ce for ce in rf.critical_eccentricity])
print ("Blendmap",rf.blendmap.dtype,rf.blendmap.shape)

pyramid = np.array(rf.makePyramid(img))
for i in xrange(rf.levels):
    print ("Input Pyramid[%d]" % i, img.dtype,img.shape)
for i in xrange(rf.levels):
    plt.figure("Pyramid level=%d" % (i+1),figsize=aspect)
    plt.imshow(pyramid[i])

blended = rf.blend(pyramid,pyramid[0].shape[1]/2,pyramid[0].shape[0]/2)

plt.figure("Eccentricity map",figsize=aspect)
plt.imshow(rf.blendmap[:,:,0])
plt.colorbar(orientation='horizontal')
plt.figure("Resolution map",figsize=aspect)
plt.imshow(rf.blendmap[:,:,1])
plt.colorbar(orientation='horizontal')
plt.figure("Blend map",figsize=aspect)
plt.imshow(rf.blendmap[:,:,2])
plt.colorbar(orientation='horizontal')
plt.figure("Layer map",figsize=aspect)
plt.imshow(np.array(rf.blendmap[:,:,3],dtype=np.uint8))
plt.colorbar(orientation='horizontal')

plt.figure("Blended",figsize=aspect)
plt.imshow(blended)
plt.show()
