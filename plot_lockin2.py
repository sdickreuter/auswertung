import os
import re

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter, sawtooth
#import harminv
from numba import jit
#from plotsettings import *
from scipy import signal
from scipy import ndimage

import matplotlib.pyplot as plt

#path = '/home/sei/Spektren/p42_lockintest/'
#path = '/home/sei/Spektren/p57m_did6_zeiss/'
#path = '/home/sei/Spektren/lockintest1/'
#path = '/home/sei/Spektren/lockin/'
#path = '/home/sei/Spektren/p61_vpol_test/'
#path = '/home/sei/Spektren/lockinvh3/'
#path = '/home/sei/Spektren/lockinv4/'
#path = '/home/sei/data/lockin/'
path = '/home/sei/Nextcloud_Annika/'


#samples = ['p45m_A1_h','p45m_A1_v','p45m_B1_h','p45m_B1_v','p45m_C1_h','p45m_C1_v',]
#samples = ['p45m_A1_v','p45m_A1_v_2','p45m_B1_v','p45m_B1_v_2','p45m_C1_v','p45m_C1_v_2',]
samples = ['l2','l3','l4']


maxwl = 900
minwl = 500
plot_normal = False

@jit(nopython=True)
def lockin_filter(signal, reference):
    width = signal.shape[0]
    y = np.zeros(width)
    for ind in range(width):
        y[ind] = np.sum(signal[ind,:] * reference) / len(reference)

    return y

for sample in samples:

    savedir = path+sample+'/'



    try:
        wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
        wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
        wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
        wl, counts = np.loadtxt(open(savedir + "normal.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
        spec = (counts-bg)/(lamp-dark)
        plot_normal = True
    except:
        pass


    lockin = np.loadtxt(open(savedir + "lockin.csv", "rb"), delimiter="\t")
    wl = lockin[:,0]
    lockin = lockin[:,1:]



    mask = (wl >= minwl) & (wl <= maxwl)
    #
    # plt.plot(wl[mask], spec[mask])
    # plt.savefig(savedir + "normal.png")
    # plt.close()

    n = lockin.shape[1]
    width = lockin.shape[0]

    for ind in range(n):
       #lockin[:,ind] = (lockin[:,ind]-bg)/lamp-dark
       lockin[:, ind] = (lockin[:, ind]) / (lamp - dark)

    lockin = ndimage.median_filter(lockin, 2)

    f = 0.015

    x = np.arange(0, n)
    #ref = np.cos(2 * np.pi * x * f * 2)
    #ref2 = np.cos(2 * np.pi * x * f * 2+np.pi/2)
    ref = -sawtooth(2 * np.pi * x * f * 2, 0.5)
    ref2 = -sawtooth(2 * np.pi * x * f * 2 + np.pi / 2, 0.5)
    # ref = -sawtooth(2 * np.pi * x * f , 0.5)
    # ref2 = -sawtooth(2 * np.pi * x * f  + np.pi / 2, 0.5)

    res = np.zeros(width)
    res_phase = np.zeros(width)
    res_amp = np.zeros(width)
    x2 = np.zeros(width)
    y2 = np.zeros(width)

    x2 = lockin_filter(lockin,ref)
    y2 = lockin_filter(lockin,ref2)

    # res_phase = signal.savgol_filter(res_phase,3,1)
    res_amp = np.abs(x2 + 1j * y2)
    res_phase = np.angle(x2 + 1j * y2)
    res = res_amp * (np.pi - np.abs(res_phase))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    indices = [np.argmin(np.abs(wl-600)), np.argmin(np.abs(wl-650)), np.argmin(np.abs(wl-700)), np.argmin(np.abs(wl-750))]
    for i, ind in enumerate(indices):
        buf = lockin[ind, :]
        buf = buf - np.min(buf)
        ax.plot(x, buf / np.max(buf) + i)
        ax.plot(x, ref / ref.max() + i)
        #y = buf*ref
        #ax.plot(x, y / y.max() + i)

        #inv = harminv.invert(lockin[ind,:]*ref, fmin=0.01, fmax=0.1)#, dt=1)
        #f_ind = np.abs(inv.frequency-f).argmin()
        #print(inv.frequency[f_ind])

    # ax.plot(x, ref/np.max(ref), 'g-')
    plt.savefig(savedir+"traces.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    indices = [np.argmin(np.abs(wl-600)), np.argmin(np.abs(wl-650)), np.argmin(np.abs(wl-700)), np.argmin(np.abs(wl-750))]
    for i, ind in enumerate(indices):
        d = np.absolute(np.fft.rfft(lockin[ind, :]))
        p = np.abs(np.angle(np.fft.rfft(lockin[ind, :])))
        f = np.fft.rfftfreq(x.shape[0])
        ax.plot(f[1:], d[1:] / d[1:].max() + i)
        ax.plot(f[1:], p[1:] / p[1:].max() + i,alpha=0.5)

    #plt.axvline(x=f)
    #plt.axvline(x=f * 2)
    plt.savefig(savedir+"fft.png")
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    #res_phase = np.abs(res_phase)
    ax.plot(wl[mask],res_phase[mask]/res_phase[mask].max(),alpha=0.5)#/lamp[mask])

    ax2 = ax.twinx()
    next(ax2._get_lines.prop_cycler)
    ax2.plot(wl[mask],res[mask]/res[mask].max(),alpha = 0.5)#/lamp[mask])
    res2 = savgol_filter(res, 71, 1)
    #ax2.plot(wl[mask],res2[mask]/res[mask].max())#/lamp[mask])
    plt.savefig(savedir+"lockin.png")
    plt.close()

    print(wl[np.argmax(res2)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.axhline(0)
    #res_phase = res_phase * 180 / np.pi
    ax.plot(wl[mask],res_phase[mask]/res_phase[mask].max(),alpha=0.5)#/lamp[mask])

    ax2 = ax.twinx()
    next(ax2._get_lines.prop_cycler)
    ax2.plot(wl[mask], res_amp[mask] / res_amp[mask].max(), alpha = 0.5)#/lamp[mask])
    #res2 = savgol_filter(res_amp, 71, 1)
    #ax.plot(wl[mask], res2[mask] / res_amp[mask].max())#/lamp[mask])

    plt.savefig(savedir+"lockin_psd.png")
    plt.close()


    if plot_normal:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(wl[mask],(1-spec[mask]))
        plt.savefig(savedir+"normal.png")
        plt.close()

    res_phase = np.zeros(width)
    res_fit_phase = np.zeros(width)
    freqs = np.zeros(width)
    # for ind in np.arange(400,500):
    for ind in range(width):
        print(ind)
        fft = np.fft.rfft(lockin[ind, :], norm="ortho")
        d = np.absolute(fft)
        p = np.angle(fft)
        res_amp[ind] = d[1:].max()
        res_phase[ind] = p[d[1:].argmax() + 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.axhline(0)
    # res_phase = res_phase * 180 / np.pi
    ax.plot(wl[mask], res_phase[mask] / res_phase[mask].max(), alpha=0.5)  # /lamp[mask])

    ax2 = ax.twinx()
    next(ax2._get_lines.prop_cycler)
    ax2.plot(wl[mask], res_amp[mask] / res_amp[mask].max(), alpha=0.5)  # /lamp[mask])
    #res2 = savgol_filter(res_amp, 71, 1)
    #ax.plot(wl[mask], res2[mask] / res_amp[mask].max())#/lamp[mask])

    plt.savefig(savedir + "lockin_fft.png")
    plt.close()


    fig = plt.figure()
    f = 0.015
    phases = np.linspace(np.pi/2,0,15)
    for i in range(len(phases)):
        phase = phases[i]
        ref = sawtooth(2 * np.pi * x * f * 2+phase, 0.5)
        res = lockin_filter(lockin,ref)

        plt.plot(wl[mask],res[mask]/np.max(np.abs(res[mask]))+i/10)

    plt.savefig(savedir + "lockin_phases.png")
    plt.close()

        # ref = -sawtooth(2 * np.pi * x * f * 2, 0.5)
    # ref2 = -sawtooth(2 * np.pi * x * f * 2 + np.pi / 2, 0.5)
    #
    # res = np.zeros(width)
    # res_phase = np.zeros(width)
    # res_amp = np.zeros(width)
    # x2 = np.zeros(width)
    # y2 = np.zeros(width)
    #
    # x2 = lockin_filter(x, ref)
    # y2 = lockin_filter(x, ref2)
    #
    # # res_phase = signal.savgol_filter(res_phase,3,1)
    # res_amp = np.abs(x2 + 1j * y2)
    # res_phase = np.angle(x2 + 1j * y2)
    #
    # #res_amp /= lamp
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # #ax.axhline(0)
    # res_phase = res_phase * 180/np.pi
    # p1 = ax.plot(wl[mask], res_amp[mask], alpha=0.5)  # /lamp[mask])
    # res2 = signal.savgol_filter(res_amp, 71, 1)
    # p2 = ax.plot(wl[mask], res2[mask])  # /lamp[mask])
    # ax.tick_params('y', colors=p2[0].get_color())
    # ax2 = ax.twinx()
    # next(ax2._get_lines.prop_cycler)
    # next(ax2._get_lines.prop_cycler)
    # p3 = ax2.plot(wl[mask], res_phase[mask], alpha=0.5)  # /lamp[mask])
    # ax2.tick_params('y', colors=p3[0].get_color())
    #
    # ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
    # ax.patch.set_visible(False)  # hide the 'canvas'
    #
    # plt.savefig(savedir + "lockin_nopsd.png",dpi=300)
    # plt.close()