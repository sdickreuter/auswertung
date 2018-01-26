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



class GetxIndices:

    def __init__(self,x,y):
        #figWH = (8,5) # in
        self.fig = plt.figure()#figsize=figWH)
        self.ax = self.fig.add_subplot(111)
        self.x = x
        self.y = y
        self.ax.plot(x,y)
        self.indices = []
        self.lines = [] # will contain 2D line objects for each of 4 lines

        self.cursor = mwidgets.Cursor(self.ax, useblit=True, color='k')
        self.cursor.horizOn = False

        self.connect = self.ax.figure.canvas.mpl_connect
        self.disconnect = self.ax.figure.canvas.mpl_disconnect

        self.clickCid = self.connect("button_press_event",self.onClick)

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return int(idx)

    def onClick(self, event):
        if event.button == 1:
            if event.inaxes:
                print(self.x)
                print(event.xdata)
                ind = self.find_nearest(self.x,event.xdata)
                self.indices.append(ind)
                self.ax.scatter(x[ind],y[ind],color="Red",marker="x",zorder=100,s=50)
                self.fig.canvas.draw()
        else:
            self.cleanup()

    def cleanup(self):
        self.disconnect(self.clickCid)
        plt.close()



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
        filtered = savgol_filter(counts, 51, 0, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        mask = ((wl < 480) & (wl > 440) ) | ((wl > 800) & (wl < 1000) )

        x = wl[mask]
        y = counts[mask]
        fit_fun = lambda x, a,x0,sigma,c: lorentz(x,a,x0,sigma)+c
        p0 = [y.max()*3,360,100,0.001]

        # plt.plot(x,y,linestyle='',marker='.')
        # plt.plot(x,fit_fun(x,p0[0],p0[1],p0[2],p0[3]),linestyle='',marker='.')
        # plt.show()
        # plt.close()

        popt, pcov = curve_fit(fit_fun, x, y, p0)
        #
        #
        # x = wl
        # y = counts-fit_fun(x,popt[0],popt[1],popt[2],popt[3])

        x = wl
        y = counts

        mask = ((wl > 450) & (wl < 820))
        x = x[mask]
        y = y[mask]

        xIndices = GetxIndices(x,y)
        plt.show()
        plt.close()
        indices = xIndices.indices


        amps = y[indices]/2
        x0s = x[indices]
        sigmas = np.repeat(10,len(indices))

        amps = np.hstack((amps,popt[0]))
        x0s = np.hstack((x0s, popt[1]))
        sigmas = np.hstack((sigmas, popt[2]))

        p0 = np.hstack((amps, x0s, sigmas,0.0))
        n = int(len(amps))

        upper = np.hstack((np.repeat(max(y), n), np.repeat(max(wl), n), np.repeat(1000, n),1))#+np.inf))
        lower = np.hstack((np.repeat(0, n), np.repeat(0, n), np.repeat(1, n),-1))#-np.inf))
        bounds = [lower,upper]


        try:
            popt, pcov = curve_fit(lorentzSum, x, y, p0,bounds=bounds,max_nfev=1000*len(x))
            perr = np.sqrt(np.diag(pcov))
            print(popt)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            amps, x0s, sigmas, c = separate_parameters(popt)
            amps_err, x0s_err, sigmas_err, c_err = separate_parameters(perr)


            for j in range(len(amps)):
                ax.plot(x, lorentz(x, amps[j], x0s[j], sigmas[j]) )#+ c)

            ax.plot(x, y, linestyle='', marker='.')
            y_fit = lorentzSum(x, *popt)
            ax.plot(x, y_fit,color='black')

            indices = np.zeros(len(amps),dtype=np.int)
            for j in range(len(x0s)):
                indices[j] = (np.abs(x - x0s[j])).argmin()
            print(indices)

            texts = []
            for j in range(len(indices)):
                texts.append(ax.text(x[indices[j]],y_fit[indices[j]],str(int(round(x0s[j])))))

            adjust_text(texts, x,y, only_move={'points':'y', 'text':'y'},
                        arrowprops=dict(arrowstyle="->", color='k', lw=1),
                        expand_points=(1.7, 1.7),
                        )#force_points=0.1)

            plt.show()

            plt.ylabel(r'$I_{scat}\, /\, a.u.$')
            plt.xlabel(r'$\lambda\, /\, nm$')
            plt.tight_layout()
            plt.savefig(savedir + "fitted/" + file[:-4] + ".png")
            #plt.show()
            plt.close()



            f = open(savedir + "fitted/" + file[:-4] + ".csv", 'w')
            f.write("x0,x0_err,amp,amp_err,sigma,sigma_err" + "\r\n")
            for j in range(len(amps)):
                f.write(str(x0s[j])+','+str(x0s_err[j])+','+str(amps[j])+','+str(amps_err[j])+','+str(sigmas[j])+','+str(sigmas_err[j])+"\r\n")

            f.close()



        except (RuntimeError,ValueError) as e:
            print(e)
            print(file + ', could not fit three lorentzians, probably not a dimer.')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, y_fit)
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