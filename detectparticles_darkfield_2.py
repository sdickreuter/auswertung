__author__ = 'sei'

import os
import re

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter

from plotsettings import *

import matplotlib.pyplot as plt
from gauss_detect import *
from gridfit import fit_grid
from make_plotmap_darkfield import make_plotmap

path = '/home/sei/Spektren/'
#samples = ['p52m_dif0','p52m_dif1','p52m_dif2','p52m_dif3','p52m_dif5']

#samples = ['p41m_dif0_ppol','p41m_dif1_ppol','p41m_dif2_ppol','p41m_dif3_ppol','p41m_dif4_ppol','p41m_dif5_ppol','p41m_dif6_ppol']
#samples = ['p41m_dif0_ppol']
#samples = ['p57m_did6_par']
#samples = ['p41m_dif3_par_3']
#samples = ['p52m_dif0_par']

#samples = ['p45m_did5_par7']

path = '/home/sei/Spektren/2C1/'
samples = ['2C1_75hept_B2']


#path = '/home/sei/Spektren/rods5/'
##samples = ['rods5_D0m','rods5_D1m','rods5_D1mm','rods5_D0mm','rods5_D1mmm','rods5_D0mmm']
#samples = ['rods5_D1mm']


# grid dimensions
nx = 5
ny = 5
maxwl = 1000
minwl = 390

for sample in samples:
    print(sample)

    savedir = path + sample + '/'

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
    plt.savefig(savedir + "plots/lamp.png")
    plt.close()
    plt.plot(wl[mask], bg[mask])
    plt.savefig(savedir + "plots/bg.png")
    plt.close()
    plt.plot(wl[mask], dark[mask])
    plt.savefig(savedir + "plots/dark.png")
    plt.close()

    #files, ids, xy = fit_grid(savedir, nx, ny)
    files, ids, xy = make_plotmap(savedir, nx, ny, minwl, maxwl,switch_xy=True)

    fig = newfig(0.9)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
    colors = cmap(np.linspace(0.1, 1, len(files)))

    meanspec = np.zeros(lamp.shape)

    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        #wl = wl[mask]
        counts = (counts - bg) / (lamp - dark)
        # counts = (counts-dark)/(lamp-dark)
        meanspec += counts
        #counts = counts[mask]
        # plt.plot(wl, counts,color=colors[i],linewidth=0.6)
        plt.plot(wl[mask], counts[mask], linewidth=0.6)

    meanspec /= len(files)
    plt.plot(wl[mask], meanspec[mask], color="black", linewidth=1)
    plt.xlim((minwl, maxwl))
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$\lambda [nm]$')
    plt.tight_layout()
    plt.savefig(savedir + 'Overview' + ".png", dpi=200)
    plt.close()

    n = xy.shape[0]

    class Resonances():
        def __init__(self, id, amp, x0, sigma):
            self.id = id
            self.amp = amp
            self.x0 = x0
            self.sigma = sigma

    int532 = np.zeros(n)
    int581 = np.zeros(n)
    peak_wl = np.zeros(n)
    max_wl = np.zeros(n)
    resonances = np.array(np.repeat(None,n),dtype=object)
    searchmask = (wl >= 500) & (wl <= 900)

    #nxy = [[round(x,6),round(y,6)] for x,y in xy]
    for i in range(n):
        x = xy[i, 0]
        y = xy[i, 1]
        file = files[i]
        print(ids[i]+" "+file)
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        counts = (counts - bg) / (lamp - dark)

        counts[np.where(counts == np.inf)] = 0.0
        ind = np.argmax(counts)
        max_wl[i] = wl[ind]
        #counts = (counts - bg)
        #counts = counts - np.mean(counts[950:1023])
        #l = lamp - dark
        #l = l - np.mean(l[0:50])
        #counts = counts/l
        #counts = counts / lamp
        filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        #plt.figure(figsize=(8, 6))
        newfig(0.9)
        plt.plot(wl[mask], counts[mask],linewidth=1)
        #plt.plot(wl[mask], filtered[mask],color="black",linewidth=0.6)
        plt.ylabel(r'$I_{df}\, /\, a.u.$')
        plt.xlabel(r'$\lambda\, /\, nm$')
        plt.xlim((minwl, maxwl))
        #plt.plot(wl, counts)
        plt.tight_layout()
        #plt.savefig(savedir + "plots/" + ids[i] + ".png",dpi=300)
        plt.savefig(savedir + "plots/" + ids[i] + ".eps",dpi=1200)
        plt.close()
        #xx = wl[330:700]
        #yy = counts[330:700]
        #wl = wl[300:900]
        #counts = counts[300:900]
        xx = wl[searchmask][::5]
        #yy = counts[searchmask]
        yy = filtered[searchmask][::5]
        wl = wl[mask]
        counts = counts[mask]
        filtered = filtered[mask]
        #min_height = max(counts)*0.1#np.mean(counts[20:40])
        min_height = 0.01#0.006#0.015
        #----amp, x0, sigma = findGausses(xx,yy,min_height,30)
        #plotGausses(savedir + "plots/" + ids[i] + "_gauss.png",wl, filtered, amp, x0, sigma)
        #amp, x0, sigma = fitLorentzes(xx, yy, amp, x0, sigma,min_height*5,2,200)
        #def fitLorentzes_iter(x, y, min_heigth, min_sigma, max_sigma, iter):
        #amp, x0, sigma = fitLorentzes_iter(xx,yy,min_height,2,100,5)
        #amp, x0, sigma = fitLorentzes2(xx, yy, min_height,5,180,max_iter=len(amp)+1)


    #     #----amp, x0, sigma = fitLorentzes3(xx, yy, amp, x0, sigma,min_height,2,200)
    #
    #     amp, x0, sigma = fitLorentzes_iter(xx, yy, min_height,5,200, 2)
    #
    #
    #     #amp, x0, sigma = findLorentzes(xx,yy,min_height,100)
    #
    #
    #     ###plotLorentzes(savedir + "plots/" + ids[i] + "_lorentz_test.png",wl, counts, amp, x0, sigma)
    #     #amp, x0, sigma = fitLorentzes(xx, yy, amp, x0, sigma,min_height,20,500)
    #
    #     #plotLorentzes(savedir + "plots/" + ids[i] + "_lorentz.png",wl, counts, amp, x0, sigma)
    #     plotLorentzes(savedir + "plots/" + ids[i] + "_lorentz.png", wl, filtered, amp, x0, sigma)
    #
    #     #print(sigma)
    #     f = open(savedir + "specs/" + ids[i] + ".csv", 'w')
    #     f.write("x" + "\r\n")
    #     f.write(str(x) + "\r\n")
    #     f.write("y" + "\r\n")
    #     f.write(str(y) + "\r\n")
    #     f.write("\r\n")
    #     f.write("wavelength,intensity" + "\r\n")
    #     for z in range(len(counts)):
    #         f.write(str(wl[z]) + "," + str(counts[z]) + "\r\n")
    #     f.close()
    #     ind = np.min(np.where(wl >= 531.5))
    #     int532[i] = counts[ind]
    #     ind = np.min(np.where(wl >= 580.5))
    #     int581[i] = counts[ind]
    #     ###resonances[i] = Resonances(ids[i],amp,x0,sigma)
    #     ###if len(x0) > 0:
    #     ###    peak_wl[i] = x0[0]
    #     ###else:
    #     ###    #peak_wl[i] = 0
    #     ###    peak_wl[i] = wl[np.argmax(filtered)]
    #
    #
    # f = open(savedir+"peaks_532nm.csv", 'w')
    # f.write("x,y,id,max"+"\r\n")
    # for i in range(len(ids)):
    #     f.write(str(xy[i,0])+","+str(xy[i,1])+","+str(ids[i])+","+str(int532[i])+"\r\n")
    # f.close()
    #
    # f = open(savedir+"peaks_581nm.csv", 'w')
    # f.write("x,y,id,max"+"\r\n")
    # for i in range(len(ids)):
    #     f.write(str(xy[i,0])+","+str(xy[i,1])+","+str(ids[i])+","+str(int581[i])+"\r\n")
    # f.close()
    #
    # f = open(savedir+"peak_wl.csv", 'w')
    # f.write("x,y,id,peak_wl"+"\r\n")
    # for i in range(len(ids)):
    #     f.write(str(xy[i,0])+","+str(xy[i,1])+","+str(ids[i])+","+str(peak_wl[i])+"\r\n")
    # f.close()
    #
    # import pickle
    #
    # with open(savedir+r"resonances.pickle", "wb") as output_file:
    #     pickle.dump(resonances, output_file)
    #
    # # newfig(0.9)
    # # plt.hist(peak_wl,50)
    # # plt.xlabel(r'$\lambda_{max}\, /\, nm$')
    # # plt.ylabel('Häufigkeit')
    # # # plt.plot(wl, counts)
    # # plt.tight_layout()
    # # plt.savefig(savedir + "hist.png", dpi=300)
    # # plt.close()