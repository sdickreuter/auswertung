import os
import re

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter

#from plotsettings import *

import matplotlib.pyplot as plt

#path = '/home/sei/Spektren/p42_lockintest/'
#path = '/home/sei/Spektren/p57m_did6_zeiss/'
path = '/home/sei/Spektren/lockintest1/'

maxwl = 1000
minwl = 500

savedir = path


wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
# wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
# wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
# wl, counts = np.loadtxt(open(savedir + "normal.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
#
#
# spec = (counts-bg)/(lamp-dark)
mask = (wl >= minwl) & (wl <= maxwl)
#
# plt.plot(wl[mask], spec[mask])
# plt.savefig(savedir + "normal.png")
# plt.close()


lockin = np.loadtxt(open(savedir + "lockin.csv", "rb"), delimiter="\t")
#lockin = lockin[:,1:]


f = 0.03

res1 = np.zeros(lockin.shape[0])
for i in range(lockin.shape[0]):
    x = np.arange(0, lockin.shape[1])
    ref = np.cos(2 * np.pi * x * f)
    buf = ref * lockin[i, :]
    buf = np.sum(buf)
    res1[i] = -buf

#plt.plot(wl[mask],  ((res1)/(lamp))[mask])
plt.plot(wl[mask],  res1[mask])
plt.savefig(savedir + "lockin_standard.png")
plt.close()


def cos_fit(x, amplitude, offset):
    return (np.cos(2 * np.pi * x * f + 0)) ** 2 * amplitude + offset

res2 = np.zeros(lockin.shape[0])
for i in range(lockin.shape[0]):
    x = np.arange(0, lockin.shape[1])
    ref = np.cos(2 * np.pi * x * f)
    buf = ref * lockin[i, :]
    p0 = [np.max(buf)-np.min(buf),np.mean(buf)]
    popt, pcov = curve_fit(cos_fit, x, buf, p0=p0)
    res2[i] = -popt[0]

plt.plot(wl[mask], ((res2)/(lamp))[mask])
plt.savefig(savedir + "lockin_fit.png")
plt.close()


# lockincorr = np.zeros(lockin.shape)
# for i in range(lockincorr.shape[1]):
#     lockincorr[:,i] = (lockin[:,i]-bg)/(lamp-dark)
#
# res3 = np.zeros(lockincorr.shape[0])
# for i in range(lockincorr.shape[0]):
#     x = np.arange(0, lockincorr.shape[1])
#     ref = np.cos(2 * np.pi * x * f)
#     buf = ref * lockincorr[i, :]
#     buf = np.sum(buf)
#     res3[i] = -buf
#
# plt.plot(wl[mask], res3[mask])
# plt.savefig(savedir + "lockincorr_standard.png")
# plt.close()


# def cos_fit(x, amplitude, offset):
#     return (np.cos(2 * np.pi * x * f + 0)) ** 2 * amplitude + offset
#
# lockincorr[np.isneginf(lockincorr)] = 0
# lockincorr[np.isinf(lockincorr)] = 0
#
# res4 = np.zeros(lockincorr.shape[0])
# for i in range(lockincorr.shape[0]):
#     x = np.arange(0, lockincorr.shape[1])
#     ref = np.cos(2 * np.pi * x * f)
#     buf = ref * lockincorr[i, :]
#     p0 = [np.max(buf) - np.min(buf), np.mean(buf)]
#     popt, pcov = curve_fit(cos_fit, x, buf, p0=p0)
#     res4[i] = -popt[0]
#     #res4[i] = -np.sum(cos_fit(x,*popt))
#
#
# plt.plot(wl[mask], res4[mask])
# plt.savefig(savedir + "lockincorr_fit.png")
# plt.close()





#
# newfig(0.9)
# plt.plot(wl[mask], counts[mask], linewidth=1)
# plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.6)
# plt.ylabel(r'$I_{df}\, /\, a.u.$')
# plt.xlabel(r'$\lambda\, /\, nm$')
# # plt.plot(wl, counts)
# plt.tight_layout()
# plt.savefig(savedir + "plots/" + files[i] + ".png", dpi=300)
# plt.close()