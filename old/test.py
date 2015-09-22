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

img = np.asarray(Image.open(pkg_resources.resource_filename("gazetools", "resources/images/waldo.png")))
# img = np.asarray(Image.open(pkg_resources.resource_filename("gazetools", "resources/images/PM5544_with_non-PAL_signals.png")))
# plt.figure("Original",figsize=(16.80,10.50))
# plt.imshow(img)

rf = RetinaFilter(ctx,1680,1050,473.76,296.1,700.0)

pyramid = np.array(rf.makePyramid(img))
for i in xrange(rf.levels):
    plt.figure("Pyramid level=%d" % (i+1),figsize=(16.80,10.50))
    plt.imshow(pyramid[i])

blended = rf.blend(pyramid,pyramid[0].shape[1]/2,pyramid[0].shape[0]/2)

plt.figure("Eccentricity map",figsize=(16.80,10.50))
plt.imshow(rf.blendmap[:,:,0])
plt.figure("Resolution map",figsize=(16.80,10.50))
plt.imshow(rf.blendmap[:,:,1])
plt.figure("Blend map",figsize=(16.80,10.50))
plt.imshow(rf.blendmap[:,:,2])
plt.figure("Layer map",figsize=(16.80,10.50))
plt.imshow(np.array(rf.blendmap[:,:,3],dtype=np.uint8))

# print blended.shape
plt.figure("Blended",figsize=(16.80,10.50))
plt.imshow(blended)
# plt.colorbar(orientation='horizontal')
plt.show()
