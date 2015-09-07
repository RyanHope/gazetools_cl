#!/usr/bin/env python

import pkg_resources
import pyopencl as cl
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import convolve1d

from gazetools import *

ctx = cl.create_some_context()

smooth = savgol_coeffs(11, 2, 0, 1.0/500.0)
print smooth
df=pd.read_csv(pkg_resources.resource_filename("data","smi.csv"))
print df.info()
