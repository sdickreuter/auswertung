__author__ = 'sei'

import os
import re

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter

#from plotsettings import *

from gauss_detect import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from matplotlib import ticker
from adjustText import adjust_text
import matplotlib.widgets as mwidgets
import peakutils

path = '/home/sei/Spektren/'

samples = ['p45m_did5_par13']
#samples = ['p41m_dif5_par']


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

    last_popt = None
    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        if not is_extinction:
            counts = (counts - bg) / (lamp - dark)
        else:
            counts = 1 - (counts - dark) / (lamp - dark)

        counts[np.where(counts == np.inf)] = 0.0
        filtered = savgol_filter(counts, 71, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        x = wl
        y = counts

        mask = ((wl > 450) & (wl < 820))
        x = x[mask]
        y = y[mask]
        filtered= filtered[mask]

        indexes = peakutils.indexes(filtered, thres=0.2, min_dist=100)
        #print(pos[indexes])
        try:
            if len(indexes)>0:
                peakx = peakutils.interpolate(x, y, ind=indexes,width=20)
                peaky = y[indexes]
                #sorted_ind = np.argsort(peaky)
                sorted_ind = np.argsort(peakx)
                sorted_ind = np.flipud(sorted_ind)
                indexes = indexes[sorted_ind]
                #peakx = peakx[sorted_ind[0]]
                #peaky = peaky[sorted_ind[0]]
            else:
                raise RuntimeError()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x,y)
            ax.plot(x,filtered)


            texts = []
            for j in range(len(indexes)):
                texts.append( ax.text( x[indexes[j]],y[indexes[j]],str(int(round(x[indexes[j]])))))

            adjust_text(texts, x,y, only_move={'points':'y', 'text':'y'},
                        arrowprops=dict(arrowstyle="->", color='k', lw=1),
                        expand_points=(1.7, 1.7),
                        )#force_points=0.1)

            #plt.show()

            plt.tight_layout()
            plt.savefig(savedir + "fitted/" + file[:-4] + ".png")
            #plt.show()
            plt.close()

            f = open(savedir + "fitted/" + file[:-4] + ".csv", 'w')
            f.write("x0" + "\r\n")
            for j in range(len(indexes)):
                f.write(str(x[indexes[j]])+"\r\n")

            f.close()



        except (RuntimeError,ValueError) as e:
            print(e)
            print(file + ', could not fit three lorentzians, probably not a dimer.')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, y)
            plt.tight_layout()
            plt.savefig(savedir + "fitted/" + file[:-4] + ".png")
            #plt.show()
            plt.close()
        # newfig(0.9)
        # plt.plot(wl[mask], counts[mask], color="grey", linewidth=0.5)
        # plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.5)
        # plt.ylabel(r'$I_{df}\, /\, a.u.$')
        # plt.xlabel(r'$\lambda\, /\, nm$')
        # plt.xlim((minwl, maxwl))
        # plt.ylim([np.min(filtered[mask]), np.max(filtered[mask]) * 1.2])
        # plt.tight_layout()
        # #plt.savefig(savedir + "plots/" + file[:-4] + ".pdf", dpi=300)
        # #plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
        # plt.show()
        # plt.close()






    # #plt.tight_layout(.5)
    # gs1.update(wspace=0.1, hspace=0.1) # set the spacing between axes.
    # #plt.show()
    # ax1 = plt.subplot(gs1[nx, :])
    # m = plt.cm.ScalarMappable(cmap=cm)
    # # m.set_array(np.linspace(0,1,len(wl)))
    # m.set_array(wl[mask][::10])
    # cb = plt.colorbar(m, cax=ax1, orientation='horizontal')
    # tick_locator = ticker.MaxNLocator(nbins=5)
    # cb.locator = tick_locator
    # cb.update_ticks()
    # cb.ax.tick_params(axis='y', direction='out')
    # cb.set_label(r'$\lambda\, /\, nm$')
    #
    # plt.savefig(savedir+"overview/" + sample +"_overview.pdf", dpi= 300)
    # plt.savefig(savedir+"overview/" + sample +"_overview.pgf")
    # plt.close()