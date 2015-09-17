import numpy as np
import pandas as pd

from .filter import savgol_coeffs, convolve1d2
from .visualangle import subtended_angle2

def pva(ctx, data, samplerate, rx, ry, sw, sh, minsac=0.02, vt=1000.0, at=100000.0):
    order = 2
    ts = 1.0 / samplerate
    window = 2 * np.ceil(minsac/ts) + 3
    filters = [savgol_coeffs(window, order, d, ts) for d in xrange(order+1)]

    dummy = np.zeros(data.shape[0])
    data["xa"] = subtended_angle2(ctx, data["x"], dummy, 0, 0, rx, ry, sw, sh, data["ez"], data["ex"], data["ey"])
    data["ya"] = subtended_angle2(ctx, dummy, data["y"], 0, 0, rx, ry, sw, sh, data["ez"], data["ex"], data["ey"])

    va = convolve1d2(ctx, np.array(data[["xa","ya","xa","ya"]],dtype=np.float32), filters[1], filters[2])
    sb = convolve1d2(ctx, np.array(data[["x","y","px","py"]],dtype=np.float32), filters[0], filters[1])

    data["sx"] = sb[:,0]
    data["sy"] = sb[:,1]
    data["pvx"] = sb[:,2]
    data["pvy"] = sb[:,3]
    data["vx"] = va[:,0]
    data["vy"] = va[:,1]
    data["ax"] = va[:,2]
    data["ay"] = va[:,3]

    data["v"] = np.sqrt(data["vx"]**2 + data["vy"]**2)
    data["a"] = np.sqrt(data["ax"]**2 + data["ay"]**2)

    data["timestamp"] = np.arange(data.shape[0]) * ts

    return data
