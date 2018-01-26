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

#samples = ['p45m_did5_par13']
#samples = ['p52m_dif5_par']
#samples = ['p45m_did5_par','p45m_did6_par']
#samples = ['p41m_dif3_par','p41m_dif4_par','p41m_dif5_par']
#samples = ['ED30_1_test']
#samples = ['p41m_dif5_par1']
sample = "2C1"
arrays = ["2C1_75hept_B2"]


# grid dimensions
nx = 7
ny = 7
maxwl = 900
minwl = 400

letters = [chr(c) for c in range(65, 91)]

for a in arrays:
    print(sample)

    savedir =  path + sample+'/'+a+'/'

    try:
        os.mkdir(savedir + "overview/")
    except:
        pass



    files = []
    for file in os.listdir(savedir + 'specs/'):
        if re.fullmatch(r"([A-Z]{1}[0-9]{1})(_corr.csv)$", file) is not None:
            files.append(file)

    print(files)

    max_int = 0.0
    min_int = np.inf
    for i in range(len(files)):
        wl, counts = np.loadtxt(open(savedir + 'specs/'+files[i]), delimiter=",", skiprows=6, unpack=True)
        mask = (wl >= minwl) & (wl <= maxwl)
        filtered = savgol_filter(counts[mask], 41, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        if filtered.max() > max_int:
            max_int = filtered.max()
        if filtered.min() < min_int:
            min_int = filtered.min()


    print(max_int)
    print(min_int)

    sns.set_style("ticks", {'axes.linewidth': 1.0,
                            'xtick.direction': 'out',
                            'ytick.direction': 'out',
                            'xtick.major.size': 3,
                            'xtick.minor.size': 1.5,
                            'ytick.major.size': 3,
                            'ytick.minor.size': 1.5
                            })

    cm = plt.cm.get_cmap('rainbow')

    wl, counts = np.loadtxt(open(savedir + 'specs/A1_corr.csv', "rb"), delimiter=",", skiprows=6, unpack=True)
    mask = (wl >= minwl) & (wl <= maxwl)

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
            id = (letters[iy] + "{0:d}".format(ix+1))
            try:
                wl, counts = np.loadtxt(open(savedir + 'specs/' + id +'_corr.csv', "rb"), delimiter=",", skiprows=6, unpack=True)

                mask = (wl >= minwl) & (wl <= maxwl)

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
                #ax.set_ylim([np.min(filtered[mask]),np.max(filtered[mask])*1.2])
                ax.set_ylim([min_int, max_int*1.1])

                ax.text(minwl*1.1, max_int*0.9, id, fontsize=6)
                #ax.ylabel(r'$I_{df} [a.u.]$')
                #ax.xlabel(r'$\lambda [nm]$')
                #ax.ylabel(None)
                #ax.xlabel(None)
                #ax.tight_layout()

            except Exception as e:
                print(e)
                print(savedir + 'specs/' + id +'_corr.csv not found, trying next file')

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
    plt.savefig(savedir+"overview/" + sample +"_overview.eps",dpi=600)

    plt.close()