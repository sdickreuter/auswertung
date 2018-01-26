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


path = '/home/sei/Spektren/'

samples = ['p45m_did5_par5']

def lorentz(x, amplitude, xo, sigma):
    g = amplitude * np.power(sigma / 2, 2) / (np.power(sigma / 2, 2) + np.power(x - xo, 2))
    return g.ravel()

# https://www.webpages.uidaho.edu/brauns/vibspect1.pdf
def asymlorentz(x, amplitude, x0, sigma, asy):
    sigma = 2 * sigma/(1 + np.exp(asy*(x-x0)) )
    g = lorentz(x,amplitude,x0,sigma)
    return g.ravel()

def three_lorentz(x,c, a0,a1,a2, xo0, xo1, xo2, fwhm0, fwhm1, fwhm2):
    g = lorentz(x,a0,xo0,fwhm0)+lorentz(x,a1,xo1,fwhm1)+lorentz(x,a2,xo2,fwhm2)+c
    return g.ravel()

def three_lorentz_asy(x,c, a0,a1,a2, xo0, xo1, xo2, fwhm0, fwhm1, fwhm2,asy0,asy1,asy2,m):
    g = asymlorentz(x,a0,xo0,fwhm0,asy0)+asymlorentz(x,a1,xo1,fwhm1,asy1)+asymlorentz(x,a2,xo2,fwhm2,asy2)+c+m*x
    return g.ravel()

def two_lorentz(x,c, a0,a1, xo0, xo1, fwhm0, fwhm1):
    g = lorentz(x,a0,xo0,fwhm0)+lorentz(x,a1,xo1,fwhm1)+c
    return g.ravel()

def two_lorentz_asy(x,c, a0,a1, xo0, xo1, fwhm0, fwhm1,asy0,asy1):
    g = asymlorentz(x,a0,xo0,fwhm0,asy0)+asymlorentz(x,a1,xo1,fwhm1,asy1)+c
    return g.ravel()

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
        filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)


        mask = ((wl < 480) & (wl > 440) ) | ((wl > 800) & (wl < 900) )

        x = wl[mask]
        y = counts[mask]
        fit_fun = lambda x, a,b,c: a*np.exp(b*x)+c

        # p0 = [y.max()*3000,-0.02,0]
        # # plt.plot(x,y,linestyle='',marker='.')
        # # plt.plot(x,fit_fun(x,p0[0],p0[1],p0[2]),linestyle='',marker='.')
        # # plt.show()
        # # plt.close()
        #
        # popt, pcov = curve_fit(fit_fun, x, y, p0)
        #
        # # x= wl[wl > 450]
        # # y = counts[wl > 450]
        # # plt.plot(x,y,linestyle='',marker='.')
        # # plt.plot(x,fit_fun(x,popt[0],popt[1],popt[2]),linestyle='',marker='.')
        # # plt.show()
        # # plt.close()
        #
        # x = wl
        # y = counts-fit_fun(x,popt[0],popt[1],popt[2])


        fit_fun = lambda x, a,x0,sigma,c: lorentz(x,a,x0,sigma)+c
        p0 = [y.max()*3,360,100,0.001]

        # plt.plot(x,y,linestyle='',marker='.')
        # plt.plot(x,fit_fun(x,p0[0],p0[1],p0[2],p0[3]),linestyle='',marker='.')
        # plt.show()
        # plt.close()

        popt, pcov = curve_fit(fit_fun, x, y, p0)
        #
        # x= wl[wl > 450]
        # y = counts[wl > 450]
        # plt.plot(x,y,linestyle='',marker='.')
        # plt.plot(x,fit_fun(x,popt[0],popt[1],popt[2],popt[3]),linestyle='',marker='.')
        # plt.show()
        # plt.close()


        x = wl
        y = counts-fit_fun(x,popt[0],popt[1],popt[2],popt[3])

        # mask1 = (wl < 480) & (wl > 450)
        # mask2 = (wl > 800) & (wl < 900)

        # x = wl
        # y = counts

        mask = ((wl > 460) & (wl < 800))
        x = x[mask]
        y = y[mask]


        #def three_lorentz_asy(x, c, a0, a1, a2, xo0, xo1, xo2, fwhm0, fwhm1, fwhm2, asy0, asy1, asy2):
        #p0 = [0, y[0], y[indexes_peaks[0]], y[indexes_peaks[1]], x[0], x[indexes_peaks[0]], x[indexes_peaks[1]], 50, 50,50]
        p0 = [-0.004,  0.005, 0.005, 0.01,  515, 575, 650,  50, 50,50, 0.0001,0.0001,0.0001,  0]

        bounds = [(-np.inf,y.min(),y.min(),y.min(),  500,550,550,   10,10,10, -0.1,-0.1,-0.1,  -1),
                  (np.inf, y.max(),y.max(),y.max(),  530, 590,700,  1000,1000,1000,  0.1,0.1,0.1,  0)]

        # plt.plot(x, y, linestyle='', marker='.')
        # plt.plot(x, three_lorentz_asy(x, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7], p0[8], p0[9],p0[10],p0[11],p0[12],p0[13]),
        #          linestyle='', marker='.')
        # plt.show()
        # plt.close()

        try:
            popt, pcov = curve_fit(three_lorentz_asy, x, y, p0,bounds=bounds)
            print(popt)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(x, y, linestyle='', marker='.')
            y_fit = three_lorentz_asy(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8],
                                      popt[9],popt[10],popt[11],popt[12],popt[13])
            ax.plot(x, y_fit)
            idx1 = (np.abs(x - popt[4])).argmin()
            idx2 = (np.abs(x - popt[5])).argmin()
            idx3 = (np.abs(x - popt[6])).argmin()

            texts = []
            texts.append(ax.text(x[idx1],y_fit[idx1],str(int(round(popt[4])))))
            texts.append(ax.text(x[idx2],y_fit[idx2],str(int(round(popt[5])))))
            texts.append(ax.text(x[idx3],y_fit[idx3],str(int(round(popt[6])))))


            adjust_text(texts, x,y, only_move={'points':'y', 'text':'y'},
                        arrowprops=dict(arrowstyle="->", color='k', lw=1),
                        expand_points=(1.7, 1.7),
                        )#force_points=0.1)

            plt.tight_layout()
            plt.savefig(savedir + "fitted/" + file[:-4] + ".png")
            #plt.show()
            plt.close()

        except (RuntimeError,ValueError) as e:
            print(file + ', could not fit three lorentzians, probably not a dimer.')


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