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
img = np.asarray(Image.open(pkg_resources.resource_filename("gazetools", "resources/images/social.png")))
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
rf = RetinaFilter(ctx,img.shape[1],img.shape[0],img.dtype,vs_rx,vs_sw,vs_pd,ez)

blended = rf.filter(img,1080,img.shape[0]/2)
out = Image.frombuffer("RGBA",(img.shape[1],img.shape[0]),blended,"raw","RGBA",0,1)
out.save("blended2.png")

# plt.figure("Blended",figsize=aspect)
# plt.imshow(blended)
# plt.show()
