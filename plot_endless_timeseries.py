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


path = '/home/sei/Spektren/'

samples = ['A1_newlense_newsetup_100um']
#samples = ['series_auto']


maxwl = 900
minwl = 500

find_peaks = False

for sample in samples:
    print(sample)

    savedir = path + sample + '/'

    try:
        os.mkdir(savedir + "overview/")
    except:
        pass
    try:
        os.mkdir(savedir + "plots/")
    except:
        pass
    try:
        os.mkdir(savedir + "specs/")
    except:
        pass

    wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    #bg = dark
    wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

    mask = (wl >= minwl) & (wl <= maxwl)


    plt.plot(wl, lamp-dark)
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/lamp.pdf", dpi=300)
    plt.close()
    plt.plot(wl[mask], bg[mask])
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/bg.pdf", dpi=300)
    plt.close()
    plt.plot(wl[mask], dark[mask])
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/dark.pdf", dpi=300)
    plt.close()

    files = []
    for file in os.listdir(savedir):
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
        meta = open(savedir + file, "rb").readlines(300)
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
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        #wl = wl[mask]
        counts = (counts - bg) / (lamp - dark)
                                        #27
        filtered = savgol_filter(counts, 51, 0, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        img[:,i] = filtered[mask]

        # counts = (counts-dark)/(lamp-dark)
        #counts = counts[mask]
        # plt.plot(wl, counts,color=colors[i],linewidth=0.6)


    for i in range(img.shape[1]):
        plt.plot(wl[mask], img[:,i], linewidth=0.6)

    plt.xlim((minwl, maxwl))
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$\lambda [nm]$')
    plt.tight_layout()
    plt.savefig(savedir + 'Overview' + ".pdf", dpi=200)
    plt.close()

    cm = plt.cm.get_cmap('rainbow')
    colors = cm(np.linspace(0, 1, len(wl[mask][::3])))

    fig = newfig(0.9)
    cmap = plt.get_cmap('plasma')#sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
    colors = cmap(np.linspace(0.1, 1, len(files)))

    for i in range(img.shape[1]):
        #plt.plot(wl, img[:,i]+i*0.001,linewidth=1,color=colors[i])
        plt.plot(wl[mask], img[:, i] + i * 0.0003, linewidth=1, color=colors[i],zorder=img.shape[1]-i)

    #plt.plot(wl, meanspec[mask], color = "black", linewidth=1)
    plt.xlim((minwl, maxwl))
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$\lambda [nm]$')
    #plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir +'Overview' + ".png",dpi=300)
    plt.close()

    plt.imshow(img.T,extent=[wl.min(),wl.max(),0,len(files)],aspect=1,cmap='plasma')
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    #plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir +'img' + ".png",dpi=300)
    plt.close()

    plt.imshow(np.log(img.T),extent=[wl.min(),wl.max(),0,len(files)],aspect=1,cmap='plasma')
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    #plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir +'img_log' + ".png",dpi=300)
    plt.close()


    if find_peaks:

        peaks = peakutils.indexes(img[:, 10], thres=0.01, min_dist=300)
        #peaks_x = peakutils.interpolate(wl[mask], img[:, 0], ind=peaks,width=50)
        peaks = np.array(np.round(peaks),dtype=np.int)
        print(peaks)
        plt.plot(wl[mask],img[:, 10])
        for xc in peaks:
            plt.axvline(x=wl[mask][xc])
        plt.tight_layout()
        plt.savefig(savedir +'fit_example' + ".png",dpi=300)
        plt.close()


        maxs = np.zeros(img.shape[1])
        peaks_x = np.zeros(img.shape[1],dtype=np.int)
        peakindexes = np.zeros(img.shape[1],dtype=np.int)
        for i in range(img.shape[1]):
            peaks = peakutils.indexes(img[:,i], thres=0.01, min_dist=300)
            #peaks = np.array(np.round(peakutils.interpolate(wl[mask], img[:,i], ind=peaks,width=50)),dtype=np.int)
            peaks = peaks[peaks > 0]
            peaks = peaks[peaks < img.shape[0]]
            a = img[peaks,i]
            sorted = np.argsort(a)
            peaks = peaks[sorted]
            #print(peaks)
            #print(a[sorted])
            peakindexes[i] = peaks[-1]
            maxs[i] = img[peakindexes[i],i]

        maxs = maxs/maxs[0]
        plt.plot(dt,maxs, color = "black", linewidth=1)
        plt.ylabel(r'$I_{df} [a.u.]$')
        plt.xlabel(r'$t [min]$')
        #plt.legend(files)
        plt.tight_layout()
        plt.savefig(savedir +'trace_max' + ".png",dpi=300)
        plt.close()

        maxs = np.zeros(img.shape[1])
        for i in range(img.shape[1]):
            maxs[i] = img[(np.argmin(np.abs(wl[mask]-750))),i]

        maxs = maxs/maxs[0]
        plt.plot(dt,maxs, color = "black", linewidth=1)
        plt.ylabel(r'$I_{df} [a.u.]$')
        plt.xlabel(r't [min]$')
        #plt.legend(files)
        plt.tight_layout()
        plt.savefig(savedir +'trace' + ".png",dpi=300)
        plt.close()

        maxs = wl[peakindexes]
        plt.plot(dt, maxs, color="black", linewidth=1)
        plt.ylabel(r'$I_{df} [a.u.]$')
        plt.xlabel(r't [min]$')
        #plt.ylim([595,615])
        # plt.legend(files)
        plt.tight_layout()
        plt.savefig(savedir + 'peak_wl' + ".png", dpi=300)
        plt.close()