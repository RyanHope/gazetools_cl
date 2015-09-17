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

rf = RetinaFilter(ctx,1680,1050,473.76,296.1,700.0)
plt.figure("Eccentricity map",figsize=(16,10))
plt.imshow(rf.ecc)
# plt.figure("Resolution map",figsize=(16,10))
# plt.imshow(rf.resmap)
plt.show()
