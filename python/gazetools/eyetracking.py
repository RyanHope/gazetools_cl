import numpy as np
import pandas as pd

from .filter import savgol_coeffs, convolve1d, convolve1d2
from .visualangle import subtended_angle2

def pva(ctx, data, samplerate, rx, ry, sw, sh, minsac=0.02):
    ts = 1.0 / samplerate
    window1 = 2 * np.ceil(minsac/ts) + 1
    window2 = 2 * np.ceil(3*minsac/ts) + 1
    filters = [savgol_coeffs(window1, 3, d, ts) for d in xrange(4)]

    dummy = np.zeros(data.shape[0])
    data["xa"] = subtended_angle2(ctx, data["x"], dummy, 0, 0, rx, ry, sw, sh, data["ez"], data["ex"], data["ey"])
    data["ya"] = subtended_angle2(ctx, dummy, data["y"], 0, 0, rx, ry, sw, sh, data["ez"], data["ex"], data["ey"])

    # set pupil velocity equal to max pupil size, basically a blink will go
    # from near max to 0 to max in a short period of time (eg. < 1 second), so a blink
    # threshold of max/.5second is a fairly slow blink
    pvt = np.max(data["py"])

    va = convolve1d2(ctx, np.array(data[["xa","ya","xa","ya"]],dtype=np.float32), filters[1], filters[2])
    sj = convolve1d2(ctx, np.array(data[["x","y","xa","ya"]],dtype=np.float32), filters[0], filters[3])
    p = convolve1d(ctx, np.array(data["py"],dtype=np.float32), savgol_coeffs(window2, 2, 1, ts))

    data["sx"] = sj[:,0]
    data["sy"] = sj[:,1]
    data["pvy"] = abs(p)
    data["vx"] = va[:,0]
    data["vy"] = va[:,1]
    data["ax"] = va[:,2]
    data["ay"] = va[:,3]

    data["v"] = np.sqrt(data["vx"]**2 + data["vy"]**2)
    data["a"] = np.sqrt(data["ax"]**2 + data["ay"]**2)

    data["class"] = np.zeros(data.shape[0], dtype=np.uint32)

    data.loc[data['v'] > 75, "class"] = 2 # saccade

    data.loc[data['pvy'] > pvt, "class"] = 1 # blinks
    data.loc[data['py'] == 0, "class"] = 1 # blinks

    data["timestamp"] = np.arange(data.shape[0]) * ts

    return data
