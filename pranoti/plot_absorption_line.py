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

try:
    import cPickle as pickle
except ImportError:
    import pickle

#path = '/home/sei/Spektren/pranoti/'
path = '/home/sei/Spektren/michael/'

#samples = ['E 10283 B1 0.5s line']
#samples = ['E 10287 D6 2s line','E 10287 A2 5s line','E 10284 D7 10s line','E 10283 B2 0.5s line']
#samples = ['E10287 A1 linescan']
samples = ['2908 1s line','2908_2 2s line','2900_E45 0s line']
#samples = ['2900_E45']



minwl = 400
maxwl = 1000


letters = [chr(c) for c in range(65, 91)]

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

    mask = (wl >= minwl) & (wl <= maxwl)

    sns.set_style("ticks", {'axes.linewidth': 1.0,
                            'xtick.direction': 'in',
                            'ytick.direction': 'in',
                            'xtick.major.size': 3,
                            'xtick.minor.size': 1.5,
                            'ytick.major.size': 3,
                            'ytick.minor.size': 1.5
                            })

    plt.plot(wl, lamp-dark)
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/lamp.pdf", dpi=300)
    plt.close()

    plt.plot(wl[mask], dark[mask])
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/dark.pdf", dpi=300)
    plt.close()

    files = []
    times = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([A-Z]{1,2}[0-9]{1,2})(.csv)$", file) is not None:
            files.append(file)

    print(files)

    # for i in range(len(files)):
    #     file = files[i]
    #     wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
    #     counts = (counts - dark) / (lamp - dark)
    #
    #     counts[np.where(counts == np.inf)] = 0.0
    #     filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    #     newfig(0.9)
    #     plt.plot(wl[mask], counts[mask], color="grey", linewidth=0.5)
    #     plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.5)
    #     plt.ylabel(r'$transmission$')
    #     plt.xlabel(r'$\lambda\, /\, nm$')
    #     plt.xlim((minwl, maxwl))
    #     #plt.ylim([np.min(filtered[mask]), np.max(filtered[mask]) * 1.2])
    #     #plt.ylim([0, 1])
    #     plt.tight_layout()
    #     plt.savefig(savedir + "plots/" + file[:-4] + ".pdf", dpi=300)
    #     #plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
    #     plt.close()
    # #
    # #     f = open(savedir + "specs/" + file[:-4] + ".csv", 'w')
    # #     f.write("wavelength,intensity" + "\r\n")
    # #     for z in range(len(counts)):
    # #         f.write(str(wl[z]) + "," + str(counts[z]) + "\r\n")
    # #     f.close()


    x = np.zeros((len(files)))
    y = np.zeros((len(files)))
    for i in range(len(files)):
        file = files[i]
        meta = open(savedir + file, "rb").readlines(300)
        x[i] = float(meta[11].decode())
        y[i] = float(meta[13].decode())

    y -= y.min()
    x -= x.min()

    if np.diff(x).max() < np.diff(y).max():
        x = y

    sorted = np.argsort(x)
    x= x[sorted]
    files = np.array(files)
    files = files[sorted]

    # first measurement as reference !
    wl, lamp = np.loadtxt(open(savedir + files[0], "rb"), delimiter=",", skiprows=16, unpack=True)

    print(x.min(),x.max())

    nx = x.shape[0]
    absorption = np.zeros(nx)
    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        absorption_buf = 1 - np.sum((counts[mask] - dark[mask])) / np.sum((lamp[mask] - dark[mask]))
        absorption[i] = absorption_buf

    sorted = np.argsort(x)
    x = x[sorted]
    absorption = absorption[sorted]
    transmittance = 1 - absorption

    plt.plot(x,transmittance)
    plt.xlabel(r'$x\, /\, \mu m$')
    plt.ylabel(r'$T^{rel}_{'+str(minwl)+'-'+str(maxwl)+r'\,nm}$')
    plt.ylim((0,1.3))
    plt.tight_layout()
    plt.savefig(savedir + "overview/" +sample+ ".pdf", dpi=300)
    plt.savefig(path +sample+ ".png", dpi=1200)
    plt.close()

    with open(path + sample + '.pkl', 'wb') as fp:
        pickle.dump((x,transmittance), fp)


