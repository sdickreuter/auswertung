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


path = '/home/sei/Spektren/'

samples = ['p45m_did5_heat']
#samples = ['p52m_dif5_par']
#samples = ['p45m_did5_par','p45m_did6_par']
#samples = ['p41m_dif3_par','p41m_dif4_par','p41m_dif5_par']
#samples = ['ED30_1_test']
#samples = ['p41m_dif5_par1']


# grid dimensions
nx = 5
ny = 5
maxwl = 1000
minwl = 400

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
    #bg = dark

    is_extinction = False
    try :
        wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    except:
        is_extinction = True

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
    if not is_extinction:
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
        if re.fullmatch(r"([A-Z]{1}[0-9]{1})(.csv)$", file) is not None:
            files.append(file)

    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        #wl = wl[mask]
        if not is_extinction:
            counts = (counts - bg) / (lamp - dark)
        else:
            counts = 1 - (counts - dark) / (lamp - dark)

        filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        # counts = (counts-dark)/(lamp-dark)
        #counts = counts[mask]
        # plt.plot(wl, counts,color=colors[i],linewidth=0.6)
        plt.plot(wl[mask], filtered[mask], linewidth=0.6)

    plt.xlim((minwl, maxwl))
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$\lambda [nm]$')
    plt.tight_layout()
    plt.savefig(savedir + 'Overview' + ".pdf", dpi=200)
    plt.close()


    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        if not is_extinction:
            counts = (counts - bg) / (lamp - dark)
        else:
            counts = 1 - (counts - dark) / (lamp - dark)

        counts[np.where(counts == np.inf)] = 0.0
        filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        newfig(0.9)
        plt.plot(wl[mask], counts[mask], color="grey", linewidth=0.5)
        plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.5)
        plt.ylabel(r'$I_{df}\, /\, a.u.$')
        plt.xlabel(r'$\lambda\, /\, nm$')
        plt.xlim((minwl, maxwl))
        plt.ylim([np.min(filtered[mask]), np.max(filtered[mask]) * 1.2])
        plt.tight_layout()
        plt.savefig(savedir + "plots/" + file[:-4] + ".pdf", dpi=300)
        #plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
        plt.close()

        f = open(savedir + "specs/" + file[:-4] + ".csv", 'w')
        f.write("wavelength,intensity" + "\r\n")
        for z in range(len(counts)):
            f.write(str(wl[z]) + "," + str(counts[z]) + "\r\n")
        f.close()

    # #size = figsize(1.0)
    # for ix in range(nx):
    #     for iy in range(ny):
    #         i = (ix)+(iy*ny)
    #         id = (letters[iy] + "{0:d}".format(ix+1))
    #         try:
    #             fig = newfig(0.9)
    #             wl, counts = np.loadtxt(open(savedir + id +'.csv', "rb"), delimiter=",", skiprows=16, unpack=True)
    #             counts = (counts - bg)/(lamp-dark)
    #             filtered = savgol_filter(counts, 41, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    #             plt.plot(wl[mask], counts[mask], color="grey", linewidth=0.3)
    #             plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.3)
    #             plt.set_xlim([minwl, maxwl])
    #             plt.set_ylim([np.min(filtered[mask]),np.max(filtered[mask])*1.2])
    #             plt.text(minwl*1.1, np.max(filtered[mask]), id, fontsize=8)
    #             plt.ylabel(r'$I_{df} [a.u.]$')
    #             plt.xlabel(r'$\lambda [nm]$')
    #             plt.tight_layout()
    #             plt.savefig(savedir + "plots/" + id + ".png", dpi=300)
    #             plt.savefig(savedir + "plots/" + id + ".pgf")
    #             plt.close()
    #
    #         except:
    #             print('file '+savedir + id +'.csv not found, trying next file')
    #             plt.close()


    sns.set_style("ticks", {'axes.linewidth': 1.0,
                            'xtick.direction': 'out',
                            'ytick.direction': 'out',
                            'xtick.major.size': 3,
                            'xtick.minor.size': 1.5,
                            'ytick.major.size': 3,
                            'ytick.minor.size': 1.5
                            })

    cm = plt.cm.get_cmap('rainbow')
    colors = cm(np.linspace(0, 1, len(wl[mask][::3])))

    size = figsize(1.0)
    fig = plt.figure(figsize=(size[0],size[0]))
    #fig.suptitle(sample)
    h_ratio = np.concatenate((np.repeat(5,nx),[1]))
    gs1 = gridspec.GridSpec(nx+1, ny,height_ratios=h_ratio)
    c=0
    for ix in range(nx):
        for iy in range(ny):
            i = (ix)+(iy*ny)
            id = (letters[iy] + "{0:d}".format(ix))
            try:
                wl, counts = np.loadtxt(open(savedir + id +'.csv', "rb"), delimiter=",", skiprows=16, unpack=True)
                if not is_extinction:
                    counts = (counts - bg) / (lamp - dark)
                else:
                    counts = 1 - (counts - dark) / (lamp - dark)
                filtered = savgol_filter(counts, 41, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
                ax = plt.subplot(gs1[c])
                plt.axis('on')
                ax.plot(wl[mask], counts[mask], color="0.75", linewidth=0.3, zorder=1)
                #ax.plot(wl[mask], filtered[mask], color="black", linewidth=0.3)
                ax.scatter(wl[mask][::3], filtered[mask][::3], s=0.66, c=colors, edgecolors='none', cmap=cm, zorder=2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_xlim([minwl, maxwl])
                ax.set_ylim([np.min(filtered[mask]),np.max(filtered[mask])*1.2])
                ax.text(minwl*1.1, np.max(filtered[mask]), id, fontsize=6)
                #ax.ylabel(r'$I_{df} [a.u.]$')
                #ax.xlabel(r'$\lambda [nm]$')
                #ax.ylabel(None)
                #ax.xlabel(None)
                #ax.tight_layout()

            except Exception as e:
                print(e)
                print('file '+savedir + id +'.csv not found, trying next file')

            #ax.imshow(img)
            #ax.set_axis_off()
            c += 1


    #plt.tight_layout(.5)
    gs1.update(wspace=0.1, hspace=0.1) # set the spacing between axes.
    #plt.show()
    ax1 = plt.subplot(gs1[nx, :])
    m = plt.cm.ScalarMappable(cmap=cm)
    # m.set_array(np.linspace(0,1,len(wl)))
    m.set_array(wl[mask][::10])
    cb = plt.colorbar(m, cax=ax1, orientation='horizontal')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.tick_params(axis='y', direction='out')
    cb.set_label(r'$\lambda\, /\, nm$')

    plt.savefig(savedir+"overview/" + sample +"_overview.pdf", dpi= 300)
    plt.savefig(savedir+"overview/" + sample +"_overview.pgf")
    plt.close()