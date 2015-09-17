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

df=pd.read_csv(pkg_resources.resource_filename("gazetools","resources/data/smi.csv"))
eyedata = df[["smi_sxl","smi_syl","smi_ezl","smi_exl","smi_eyl","smi_dxl","smi_dyl"]]
eyedata.columns = ["x","y","ez","ex","ey","px","py"]

eyedata_pva = pva(ctx, eyedata, 500, 1680, 1050, 473.76, 296.1)

spanp = np.ceil(3*.02/4/(1.0/500)) * 2 + 1
ptp = turnpoints(ctx, eyedata_pva["pvy"], span=spanp)
pp = [i for i in xrange(len(ptp)) if (ptp[i] == 1 and eyedata_pva["class"][i]==1)]
for p in pp:
    l = p - 1
    r = p + 1
    while ptp[l]!=-1:
        l -= 1
    while ptp[r]!=-1:
        r += 1
    eyedata_pva["class"][(l+1):r] = 1

span = np.ceil(.02/4/(1.0/500)) * 2 + 1
tp = turnpoints(ctx, eyedata_pva["v"], span=span)
tp = [tp[i] if eyedata_pva["class"][i]!=1 else 0 for i in xrange(len(tp))] # ignore turnpoints from blinks
sp = [i for i in xrange(len(tp)) if (tp[i] == 1 and eyedata_pva["class"][i]==2)]
for p in sp:
    l = p - 1
    r = p + 1
    while tp[l]!=-1:
        l -= 1
    while tp[r]!=-1:
        r += 1
    eyedata_pva["class"][(l+1):r] = 2

fig = plt.figure(figsize=(16,9))

# start = int(500 * 1.5)
# end = int(500 * 2.5)
start = 0
end = eyedata_pva.shape[0]

psize = [80 if (ptp[i] == 1 or ptp[i] == -1) else 10 for i in xrange(len(ptp))]
size = [80 if (tp[i] == 1 or tp[i] == -1) else 10 for i in xrange(len(tp))]


# for p in sp:
#     i = 1
#     while tp[p-i] != -1:
#         eyedata_pva["class"][p-i] = 2
#         i += 1
#     eyedata_pva["class"][p-i] = 2
#     i = 1
#     while tp[p+i] != -1:
#         eyedata_pva["class"][p+i] = 2
#         i += 1
#     eyedata_pva["class"][p+i] = 2

#eyedata_pva["class"][]

colors = ["black","magenta","red"]
color = [colors[c] for c in eyedata_pva["class"]]

a=fig.add_subplot(5,1,1)
plt.scatter(eyedata_pva["timestamp"][start:end],eyedata_pva["py"][start:end],c=color[start:end],s=psize[start:end],edgecolor="")
a.set_title("pupil y")

a=fig.add_subplot(5,1,2)
plt.scatter(eyedata_pva["timestamp"][start:end],eyedata_pva["pvy"][start:end],c=color[start:end],s=psize[start:end],edgecolor="")
a.set_title("pupil velocity")

a=fig.add_subplot(5,1,3)
#plt.scatter(eyedata_pva["timestamp"][start:end],eyedata_pva["x"][start:end],c=eyedata_pva["class"][start:end],s=size)
plt.scatter(eyedata_pva["timestamp"][start:end],eyedata_pva["sx"][start:end],c=color[start:end],s=size[start:end],edgecolor="")
a.set_title("gaze x")

a=fig.add_subplot(5,1,4)
#plt.scatter(eyedata_pva["timestamp"][start:end],eyedata_pva["y"][start:end],c=eyedata_pva["class"][start:end],s=size)
plt.scatter(eyedata_pva["timestamp"][start:end],eyedata_pva["sy"][start:end],c=color[start:end],s=size[start:end],edgecolor="")
a.set_title("gaze y")

a=fig.add_subplot(5,1,5)
plt.scatter(eyedata_pva["timestamp"][start:end],eyedata_pva["v"][start:end],c=color[start:end],s=size[start:end],edgecolor="")
a.set_title("gaze velocity")

plt.show()
