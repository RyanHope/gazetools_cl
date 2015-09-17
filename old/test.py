import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../python"))

import pkg_resources
import pyopencl as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from gazetools import *

ctx = cl.create_some_context(answers=[0,1])

df=pd.read_csv(pkg_resources.resource_filename("gazetools","resources/data/smi.csv"))
eyedata = df[["smi_sxl","smi_syl","smi_ezl","smi_exl","smi_eyl","smi_dxl","smi_dyl"]]
eyedata.columns = ["x","y","ez","ex","ey","px","py"]

eyedata_pva = pva(ctx, eyedata, 500, 1680, 1050, 473.76, 296.1)

fig = plt.figure(figsize=(16,9))

start = 0
end = eyedata_pva.shape[0]

a=fig.add_subplot(4,1,1)
plt.plot(eyedata_pva["timestamp"][start:end],eyedata_pva["pvy"][start:end],'b-')
a.set_title("pupil velocity")

a=fig.add_subplot(4,1,2)
plt.plot(eyedata_pva["timestamp"][start:end],eyedata_pva["x"][start:end],'r-',eyedata_pva["timestamp"][start:end],eyedata_pva["sx"][start:end],'b-')
a.set_title("gaze x")

a=fig.add_subplot(4,1,3)
plt.plot(eyedata_pva["timestamp"][start:end],eyedata_pva["y"][start:end],'r-',eyedata_pva["timestamp"][start:end],eyedata_pva["sy"][start:end],'b-')
a.set_title("gaze y")

a=fig.add_subplot(4,1,4)
plt.plot(eyedata_pva["timestamp"][start:end],eyedata_pva["v"][start:end],'b-')
a.set_title("gaze velocity")

plt.show()
