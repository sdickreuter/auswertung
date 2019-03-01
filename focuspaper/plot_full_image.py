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


path = '/home/sei/Nextcloud/Annika/'

#samples = ['zmiscnolense3']
samples = ['up','down']


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
    #
    # mask = (wl >= minwl) & (wl <= maxwl)
    #
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
    plt.close()

    files = []
    picfiles = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([+-]?)([0-9]{1,5})(.csv)$", file) is not None:
            files.append(file)


    zpos = []
    for f in files:
        res = re.match(r"([+-]?[0-9]{1,5})", f)
        zpos.append(res.group(0))

    zpos = np.array(zpos,dtype=np.float)
    sorted = np.argsort(zpos)
    zpos = zpos[sorted]
    files = np.array(files)
    files = files[sorted]

    minwl_ind = np.argmin(np.abs(wl-minwl))
    maxwl_ind = np.argmin(np.abs(wl-maxwl))


    max = -np.inf
    min = np.inf
    for i in range(len(files)):
        file = files[i]
        img = np.loadtxt(savedir + file)
        img = img[:-1,:]
        for j in range(img.shape[0]):
            img[j,:] = (img[j,:]-bg) / (lamp-dark)
        img = img[120:140, minwl_ind:maxwl_ind]
        max = np.max([img.max(), max])
        min = np.min([img.min(), min])

    for file in files:
        img = np.loadtxt(savedir+file)
        wl = img[-1,:]
        img = img[:-1,:]
        plt.figure(figsize=(5, 3))
        for j in range(img.shape[0]):
            img[j,:] = (img[j,:]-bg) / (lamp-dark)

        img = img[120:140,minwl_ind:maxwl_ind]

        cm = plt.cm.get_cmap('rainbow')
        colors = cm(np.linspace(0.1, 1, img.shape[0]))
        plt.figure(figsize=(5, 3))
        for j in range(img.shape[0]):
            plt.plot(wl[minwl_ind:maxwl_ind], img[j,:],color=colors[j])
        plt.tight_layout()
        plt.savefig(savedir+ 'plots/' + file[:-4] + "_waterfall.png", dpi=600)
        plt.close()


        plt.figure(figsize=(10,2))
        plt.imshow(img,extent=[wl[minwl_ind:maxwl_ind].min(),wl[minwl_ind:maxwl_ind].max(),0,img.shape[0]], cmap='plasma',vmax=max,vmin=min)
        plt.tight_layout()
        plt.savefig(savedir+ 'plots/' + file[:-4] + ".png", dpi=600)
        plt.close()

        plt.figure()

        plt.plot(wl[minwl_ind:maxwl_ind],np.sum(img,0))
        plt.tight_layout()
        plt.savefig(savedir + 'plots/' + file[:-4] + "_sum.png", dpi=600)
        plt.close()





    sumimg = np.zeros(img.shape)
    for file in files:
        img = np.loadtxt(savedir + file)
        wl = img[-1, :]
        img = img[:-1, :]
        for j in range(img.shape[0]):
            img[j,:] = (img[j,:]-bg) / (lamp-dark)
        img = img[120:140, minwl_ind:maxwl_ind]
        sumimg += img

    plt.imshow(sumimg, cmap='plasma')
    plt.tight_layout()
    plt.savefig(savedir + 'plots/' + "sumimg.png", dpi=600)
    plt.close()

    plt.figure()
    plt.plot(wl[minwl_ind:maxwl_ind],np.sum(sumimg, 0))
    plt.tight_layout()
    plt.savefig(savedir + 'plots/' + "sumimg_sum.png", dpi=600)
    plt.close()

