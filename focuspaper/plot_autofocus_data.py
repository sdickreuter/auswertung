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
import matplotlib as mpl
import seaborn as sns
import PIL
from matplotlib import ticker
import itertools
import datetime
import peakutils
from skimage import feature
import scipy.optimize as opt

#  modified from: http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m#comment33999040_21566831
def gauss2D(pos, amplitude, xo, yo, fwhm, offset):
    sigma = fwhm / 2.3548
    g = offset + amplitude * np.exp(
        -(np.power(pos[0] - xo, 2.) + np.power(pos[1] - yo, 2.)) / (2 * np.power(sigma, 2.)))
    return g.ravel()


path = '/home/sei/Nextcloud_Annika/'

#samples = ['zmiscnolense3']
samples = ['zmisc']


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

    #wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    #bg = dark
    #wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

    mask = (wl >= minwl) & (wl <= maxwl)


    # plt.plot(wl, lamp-dark)
    # plt.xlim((minwl, maxwl))
    # plt.savefig(savedir + "overview/lamp.pdf", dpi=300)
    # plt.close()
    # plt.plot(wl[mask], bg[mask])
    # plt.xlim((minwl, maxwl))
    # plt.savefig(savedir + "overview/bg.pdf", dpi=300)
    # plt.close()
    # plt.plot(wl[mask], dark[mask])
    # plt.xlim((minwl, maxwl))
    # plt.savefig(savedir + "overview/dark.pdf", dpi=300)
    # plt.close()

    files = []
    picfiles = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([0-9]{2,3})(.)([0-9]{2})(_spec.txt)$", file) is not None:
            files.append(file)
        elif re.fullmatch(r"([0-9]{2,3})(.)([0-9]{2})(.txt)$", file) is not None:
            picfiles.append(file)


    zpos = []
    for f in files:
        res = re.match(r"([0-9]{2,3})(.)([0-9]{2})", f)
        zpos.append(res.group(0))

    zpos = np.array(zpos,dtype=np.float)
    zpos -= zpos.min()
    sorted = np.argsort(zpos)
    zpos = zpos[sorted]
    files = np.array(files)
    files = files[sorted]


    zpos2 = []
    for f in picfiles:
        res = re.match(r"([0-9]{2,3})(.)([0-9]{2})", f)
        zpos2.append(res.group(0))

    zpos2 = np.array(zpos2)
    sorted = np.argsort(zpos2)
    zpos2 = zpos2[sorted]
    picfiles = np.array(picfiles)
    picfiles = picfiles[sorted]

    print(files)
    print(picfiles)
    n = len(files)

    img = np.zeros((dark[mask].shape[0],len(files)))

    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=1, unpack=True)
        #wl = wl[mask]
        #counts = (counts - bg) / (lamp - dark)
        counts = (counts - dark)

        #27
        filtered = savgol_filter(counts, 51, 0, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        img[:,i] = filtered[mask]

        # counts = (counts-dark)/(lamp-dark)
        #counts = counts[mask]
        # plt.plot(wl, counts,color=colors[i],linewidth=0.6)

    cm = plt.cm.get_cmap('rainbow')
    colors = cm(np.linspace(0.1, 1, len(files)))

    for i in range(img.shape[1]):
        y = img[:,i].copy()
        y -= y[:10].mean()
        y /= y.max()
        plt.plot(wl[mask], y, linewidth=0.6,color=colors[i],)

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

    plt.imshow(img.T,extent=[wl.min(),wl.max(),zpos.min(),zpos.max()],aspect=66,cmap='plasma')
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    #plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir +'img' + ".png",dpi=300)
    plt.close()

    plt.imshow(np.log(img.T),extent=[wl.min(),wl.max(),zpos.min(),zpos.max()],aspect=66,cmap='plasma')
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    #plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir +'img_log' + ".png",dpi=300)
    plt.close()


    amp = []
    sigma = []
    for i in range(len(picfiles)):
        file = picfiles[i]
        img = np.loadtxt(savedir +file)

        x = np.arange(img.shape[0])
        y = np.arange(img.shape[1])
        x, y = np.meshgrid(x, y)
        xdata = (x.ravel(), y.ravel())
        ydata = img.T.ravel()

        coordinates = feature.peak_local_max(img, min_distance=20, exclude_border=2)
        if len(coordinates) < 1:
            coordinates = [img.shape[1] / 2, img.shape[0] / 2]
        else:
            coordinates = coordinates[0]

        #try:
        initial_guess = ( np.max(img) - np.min(img), coordinates[0], coordinates[1], 1, np.mean(img))
        bounds = ((0, 0, 0, 0, -np.inf),
                  (np.inf, x.max(), y.max(), np.inf, np.inf))
        popt, pcov = opt.curve_fit(gauss2D, xdata, ydata, p0=initial_guess, bounds=bounds)#, method='dogbox')
        amp.append(popt[0])
        sigma.append(popt[3])
        #except (RuntimeError, ValueError) as e:
        #    print(e)

    amp = np.array(amp)
    sigma = np.array(sigma)

    plt.plot(zpos2,1/sigma)
    plt.ylabel(r'${1}\over{\sigma} [um]$')
    plt.xlabel(r'$z [um]$')
    plt.tight_layout()
    plt.savefig(savedir + 'Sigma' + ".pdf", dpi=200)
    plt.close()

    plt.plot(zpos2,amp)
    plt.ylabel(r'$Amplitude [a.u.]$')
    plt.xlabel(r'$z [um]$')
    plt.tight_layout()
    plt.savefig(savedir + 'Amp' + ".pdf", dpi=200)
    plt.close()

    max = -np.inf
    min = np.inf
    for i in range(len(picfiles)):
        file = picfiles[i]
        img = np.loadtxt(savedir +file)
        max = np.max([img.max(),max])
        min = np.min([img.min(), min])

    for i in range(len(picfiles)):
        file = picfiles[i]
        img = np.loadtxt(savedir +file)

        plt.imshow(img.T, cmap='plasma',vmax=max,vmin=min)
        plt.tight_layout()
        plt.savefig(savedir+ 'plots/' + file[:-4] + ".png", dpi=300)
        plt.close()
