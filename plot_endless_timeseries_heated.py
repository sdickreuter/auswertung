__author__ = 'sei'

import os
import re

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter

from plotsettings import *

from gauss_detect import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from matplotlib import ticker
import itertools
import datetime
import peakutils


path = '/home/sei/Spektren/heated/'

samples = ['p52m_dif5_D4','p52m_dif5_D4_2']
#samples = ['series_auto']


maxwl = 900
minwl = 500


def load_data(dir):
    wl, lamp = np.loadtxt(open(dir + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    wl, dark = np.loadtxt(open(dir + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    #bg = dark
    wl, bg = np.loadtxt(open(dir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

    mask = (wl >= minwl) & (wl <= maxwl)

    files = []
    for file in os.listdir(dir):
        if re.fullmatch(r"([0-9]{9})(.csv)$", file) is not None:
            files.append(file)


    numbers = []
    for f in files:
        numbers.append(f[:-4])

    numbers = np.array(numbers,dtype=np.int)
    sorted = np.argsort(numbers)
    files = np.array(files)
    files = files[sorted]

    n = len(files)
    xy = np.zeros([n, 2])
    inds = np.zeros(n)
    t = []

    for i in range(n):
        file = files[i]
        meta = open(dir + file, "rb").readlines(300)
        # xy[i, 0] = float(meta[11].decode())
        # xy[i, 1] = float(meta[13].decode())
        inds[i] = i
        # wl, int[i] = np.loadtxt(open(savedir+file,"rb"),delimiter=",",skiprows=12,unpack=True)
        date = [int(''.join(i)) for is_digit, i in itertools.groupby(meta[0].decode(), str.isdigit) if is_digit]
        time = [int(''.join(i)) for is_digit, i in itertools.groupby(meta[1].decode(), str.isdigit) if is_digit]
        #tn = datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
        tn = datetime.datetime(date[2], date[1], date[0], hour=time[0], minute=time[1], second=time[2], microsecond=time[3])
        t.append(tn)

    dt = np.zeros(n)
    for i in range(n-1):
        t0 = t[i]
        dt_buf = t[i+1] - t0
        dt[i+1] = dt[i]+dt_buf.total_seconds()

    dt /= 60

    img = np.zeros((lamp[mask].shape[0],len(files)))

    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(dir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        #wl = wl[mask]
        counts = (counts - bg) / (lamp - dark)
                                        #27
        filtered = savgol_filter(counts, 51, 0, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        img[:,i] = filtered[mask]

        # counts = (counts-dark)/(lamp-dark)
        #counts = counts[mask]
        # plt.plot(wl, counts,color=colors[i],linewidth=0.6)

    return wl[mask], dt, img




dir1 = path + samples[0] + '/'
wl1, dt1, img1 = load_data(dir1)

dir2 = path + samples[1] + '/'
wl2, dt2, img2 = load_data(dir2)

fig = newfig(0.9)

indices1 = [0,int(len(dt1)/2),len(dt1)-2]
for k in range(len(indices1)):
    i = indices1[k]
    plt.plot(wl1, img1[:,i]/img1[:,i].max()-0.1*k+0.6,label=str(int(round(dt1[i])))+' min')

indices2 = [0,int(len(dt2)/2),len(dt2)-2]
for k in range(len(indices2)):
    i = indices2[k]
    plt.plot(wl2, img2[:,i]/img2[:,i].max()-0.1*k+0.3,label=str(int(round(dt2[i])))+' min',linestyle='--')


plt.ylabel(r'$I_{df} [a.u.]$')
plt.xlabel(r'$\lambda [nm]$')
plt.legend()
plt.tight_layout()
#plt.savefig(path + "comp.pdf", dpi=400)
plt.savefig(path + "comp.pgf")
plt.savefig(path + "comp.png", dpi=400)
plt.close()
