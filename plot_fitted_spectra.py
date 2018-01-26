__author__ = 'sei'

import os
import re

import numpy as np
from scipy.signal import savgol_filter

from plotsettings import *

from gauss_detect import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from matplotlib import ticker
from adjustText import adjust_text

path = '/home/sei/Spektren/'

samples = ['p45m_did5_par5']
#samples = ['p41m_dif5_par']


def lorentz(x, amplitude, xo, sigma):
    g = amplitude * np.power(sigma / 2, 2) / (np.power(sigma / 2, 2) + np.power(x - xo, 2))
    return g.ravel()

def separate_parameters(p):
    n = int( (len(p)-1) / 3)
    amp = p[:n]
    x0 = p[n:2 * n]
    sigma = p[2 * n:3 * n]
    c = p[-1]
    return amp, x0, sigma,c

def lorentzSum(x,*p):
    # n = int( (len(p)-1) / 3)
    # amp = p[:n]
    # x0 = p[n:2 * n]
    # sigma = p[2 * n:3 * n]
    amp, x0, sigma, c = separate_parameters(p)
    res = lorentz(x, amp[0], x0[0], sigma[0])
    for i in range(len(amp) - 1):
        res += lorentz(x, amp[i + 1], x0[i + 1], sigma[i + 1])
    return res+c

letters = [chr(c) for c in range(65, 91)]

for sample in samples:
    print(sample)

    savedir = path + sample + '/'

    try:
        os.mkdir(savedir + "fitted/")
    except:
        pass

    wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    #bg = dark

    is_extinction = False
    try :
        wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    except:
        is_extinction = True

    files = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([A-Z]{1}[0-9]{1})(.csv)$", file) is not None:
            files.append(file)

    #files = ['A1.csv','B1.csv','C1.csv']
    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        if not is_extinction:
            counts = (counts - bg) / (lamp - dark)
        else:
            counts = 1 - (counts - dark) / (lamp - dark)

        counts[np.where(counts == np.inf)] = 0.0
        filtered = savgol_filter(counts, 51, 0, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        x = wl
        y = counts

        mask = ((wl > 450) & (wl < 820))
        x = x[mask]
        y = y[mask]

        #x0,x0_err,amp,amp_err,sigma,sigma_err
        data = np.loadtxt(savedir + "fitted/" + file[:-4] + ".csv",skiprows=1,delimiter=',')
        print(data)
        x0s = data[:,0]
        x0s_err = data[:,1]
        amps = data[:, 2]
        amps_err = data[:, 3]
        sigmas = data[:, 4]
        sigma_err = data[:, 5]
        c = data[0,6]
        c_err = data[0, 7]

        popt = np.hstack((amps,x0s,sigmas,c))

        fig, ax = newfig(0.9)

        for j in range(len(amps)):
            ax.plot(x, lorentz(x, amps[j], x0s[j], sigmas[j]))  # + c)

        ax.plot(x, y, linestyle='', marker='.',label='measured spectrum')
        y_fit = lorentzSum(x, *popt)
        ax.plot(x, y_fit, color='black',label='sum of Lorentzians')

        indices = np.zeros(len(amps), dtype=np.int)
        for j in range(len(x0s)):
            indices[j] = (np.abs(x - x0s[j])).argmin()
        print(indices)

        # texts = []
        # for j in range(len(indices)):
        #     texts.append(ax.text(x[indices[j]], y_fit[indices[j]], str(int(round(x0s[j])))))
        #
        # adjust_text(texts, x, y, only_move={'points': 'y', 'text': 'y'},
        #             arrowprops=dict(arrowstyle="->", color='k', lw=1),
        #             expand_points=(1.7, 1.7),
        #             )  # force_points=0.1)

        plt.ylabel(r'$I_{scat}\, /\, a.u.$')
        plt.xlabel(r'$\lambda\, /\, nm$')
        plt.legend()

        plt.tight_layout()
        plt.savefig(savedir + "fitted/" + file[:-4] + ".png", dpi=600)
        plt.savefig(savedir + "fitted/" + file[:-4] + ".eps", dpi=1200)
        plt.savefig(savedir + "fitted/" + file[:-4] + ".pgf", dpi=1200)
        plt.close()
