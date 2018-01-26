import os
import re

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter

from plotsettings import *

import matplotlib.pyplot as plt
from gauss_detect import *


path = '/home/sei/data/Pascal/'
samples = ['ito_vorstruktur_60_2d','ito_vorstruktur_60_2e','ito_vorstruktur_60_3d','ito_vorstruktur_60_3f',]
#samples = ['ed10shit','ed40','ed90']
#samples = ['ed_lito_10','ed_lito_90']
#samples = ['ito_vorstruktur_60_2d','ito_vorstruktur_60_2e','ito_vorstruktur_60_3d','ito_vorstruktur_60_3f','ed10shit','ed40','ed90','ed_lito_10','ed_lito_90']
#samples = ['20er ed v2','ed20_v3']
samples = ['ed40v2','ed90v2','ed10v2','ed20v4']


maxwl = 900
minwl = 450

for sample in samples:

    savedir = path + sample + '/'

    try:
        os.mkdir(savedir + "plots/")
    except:
        pass
    try:
        os.mkdir(savedir + "specs/")
    except:
        pass

    wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=12, unpack=True)
    wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=12, unpack=True)
    bg = dark
    #wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=12, unpack=True)

    mask = (wl >= minwl) & (wl <= maxwl)

    plt.plot(wl, lamp-dark)
    plt.savefig(savedir + "plots/lamp.png")
    plt.close()
    plt.plot(wl[mask], bg[mask])
    plt.savefig(savedir + "plots/bg.png")
    plt.close()
    plt.plot(wl[mask], dark[mask])
    plt.savefig(savedir + "plots/dark.png")
    plt.close()

    files = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([0-9]{5})(.csv)$", file) is not None:
            files.append(file)

    n = len(files)
    peak_wl = np.zeros(n)
    spectra = np.zeros((n,len(lamp)))

    for i in range(n):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=12, unpack=True)
        counts = (counts - bg) / (lamp - dark)

        counts[np.where(counts == np.inf)] = 0.0

        filtered = savgol_filter(counts, 31, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        # plt.figure(figsize=(8, 6))
        newfig(0.9)
        plt.plot(wl[mask], counts[mask], linewidth=1)
        plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.6)
        plt.ylabel(r'$I_{df}\, /\, a.u.$')
        plt.xlabel(r'$\lambda\, /\, nm$')
        # plt.plot(wl, counts)
        plt.tight_layout()
        plt.savefig(savedir + "plots/" + files[i] + ".png", dpi=300)
        plt.close()

        #counts = counts[mask]
        #filtered = filtered[mask]

        # print(sigma)
        # f = open(savedir + "specs/" + files[i] + "_corr.csv", 'w')
        # f.write("wavelength,intensity" + "\r\n")
        # for z in range(len(counts)):
        #     f.write(str(wl[z]) + "," + str(counts[z]) + "\r\n")
        # f.close()
        #if len(x0) > 0:
        #    peak_wl[i] = x0[0]
        #else:
        #    # peak_wl[i] = 0
        peak_wl[i] = wl[np.argmax(filtered)]
        spectra[i,:] = counts

    newfig(0.9)
    for i in range(spectra.shape[0]):
        plt.plot(wl[mask], spectra[i,mask], linewidth=0.75, color = "lightgrey")

    plt.plot(wl[mask], np.mean(spectra[:,mask],0), color="black", linewidth=1)
    plt.ylabel(r'$I_{df}\, /\, a.u.$')
    plt.xlabel(r'$\lambda\, /\, nm$')
    # plt.plot(wl, counts)
    plt.tight_layout()
    plt.savefig(savedir + "overview.png", dpi=300)
    plt.close()


    f = open(savedir + "peak_wl.csv", 'w')
    f.write("x,y,id,peak_wl" + "\r\n")
    for i in range(len(files)):
        f.write(str(files[i]) + "," + str(peak_wl[i]) + "\r\n")
    f.close()
