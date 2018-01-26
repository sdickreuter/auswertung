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
import itertools
import datetime
import peakutils
from adjustText import adjust_text


path = '/home/sei/Spektren/heated/'

#samples = ['p52m_dif5_D4_2']
#samples = ['p45m_did5_d0']
samples = ['p52m_dif5_D4']
#samples = ['series_auto']

plot_fits = False


maxwl = 1000
minwl = 450


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

    mask = (wl >= minwl) & (wl <= maxwl)


    plt.plot(wl, lamp-dark)
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/lamp.pdf", dpi=600)
    plt.close()
    plt.plot(wl[mask], bg[mask])
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/bg.pdf", dpi=600)
    plt.close()
    plt.plot(wl[mask], dark[mask])
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/dark.pdf", dpi=600)
    plt.close()

    files = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([0-9]{9})(.csv)$", file) is not None:
            files.append(file)


    numbers = []
    for f in files:
        numbers.append(f[:-4])

    numbers = np.array(numbers,dtype=np.int)
    sorted = np.argsort(numbers)
    files = np.array(files)
    files = files[sorted]

    n = len(files)
    xy = np.zeros([n, 2])
    inds = np.zeros(n)
    t = []

    for i in range(n):
        file = files[i]
        meta = open(savedir + file, "rb").readlines(300)
        # xy[i, 0] = float(meta[11].decode())
        # xy[i, 1] = float(meta[13].decode())
        inds[i] = i
        # wl, int[i] = np.loadtxt(open(savedir+file,"rb"),delimiter=",",skiprows=12,unpack=True)
        date = [int(''.join(i)) for is_digit, i in itertools.groupby(meta[0].decode(), str.isdigit) if is_digit]
        time = [int(''.join(i)) for is_digit, i in itertools.groupby(meta[1].decode(), str.isdigit) if is_digit]
        #tn = datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
        tn = datetime.datetime(date[2], date[1], date[0], hour=time[0], minute=time[1], second=time[2], microsecond=time[3])
        t.append(tn)

    dt = np.zeros(n)
    for i in range(n-1):
        t0 = t[i]
        dt_buf = t[i+1] - t0
        dt[i+1] = dt[i]+dt_buf.total_seconds()

    dt /= 60

    sorted = np.argsort(dt)
    dt = dt[sorted]
    files = files[sorted]

    img = np.zeros((lamp[mask].shape[0],len(files)))

    for i in range(len(files)):
        file = files[i]

        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        #wl = wl[mask]
        counts = (counts - bg) / (lamp - dark)
                                        #27
        filtered = savgol_filter(counts, 27, 0, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        img[:,i] = filtered[mask]


        #if i == 7:
        #    wl, bg = np.loadtxt(open(savedir + "background2.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

    fig = newfig(0.9)
    cmap = plt.get_cmap('plasma')#sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
    colors = cmap(np.linspace(0.1, 1, len(files)))

    for i in range(img.shape[1]):
        plt.plot(wl[mask], img[:,i], color=colors[i], linewidth=0.6)

    plt.xlim((minwl, maxwl))
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$\lambda [nm]$')
    plt.tight_layout()
    plt.savefig(savedir + 'Overview' + ".pdf", dpi=600)
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
    plt.savefig(savedir +'Overview' + ".png",dpi=600)
    plt.close()


    img2 = img.copy()
    for i in range(img.shape[1]):
        img2[:,i] = img[:,i]/img[:,i].max()

    plt.imshow(img2.T,extent=[wl.min(),wl.max(),0,len(files)],aspect=1,cmap='plasma')
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    #plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir +'img' + ".png",dpi=600)
    plt.close()

    plt.imshow(np.log(img2.T),extent=[wl.min(),wl.max(),0,len(files)],aspect=1,cmap='plasma')
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    #plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir +'img_log' + ".png",dpi=600)
    plt.close()


    indices = [0,int(len(files)/2),len(files)-1]

    for i in indices:
        plt.plot(wl[mask],img[:, i],label = str(int(dt[i]))+' min')
    plt.legend()
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    plt.tight_layout()
    plt.savefig(savedir +'spectrum_comparison.png',dpi=600)
    plt.close()

    for i in indices:
        plt.plot(wl[mask],img[:, i]/img[:,i].max(),label = str(int(dt[i]))+' min')
    plt.legend()
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    plt.tight_layout()
    plt.savefig(savedir +'spectrum_comparison_norm.png',dpi=600)
    plt.close()


    #indices = [0,1,2,3,5,6,7,150,151,152,153,154,155,156,157,160,175,176,177,178,179,180,181,182,183]
    indices = np.hstack( (np.arange(0,7,1), np.arange(int(len(files)/2),int(len(files)/2)+7,1), np.arange(len(files)-8,len(files)-1,1)  )  )

    cm = plt.cm.get_cmap('rainbow')
    cm = cm(np.linspace(0,1,len(indices)))
    for i,index in enumerate(indices):
        plt.plot(wl[mask],img[:, index],color=cm[i])
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    plt.tight_layout()
    plt.savefig(savedir +'spectrum_comparison_more.png',dpi=600)
    plt.close()

    for i,index in enumerate(indices):
        plt.plot(wl[mask],img[:, index]/img[:,index].max(),color=cm[i])
    plt.xlabel(r'$\lambda [nm]$')
    plt.ylabel(r'$z [\mu m]$')
    plt.tight_layout()
    plt.savefig(savedir +'spectrum_comparison_norm_more.png',dpi=600)
    plt.close()


    peaks = peakutils.indexes(img[:, 10], thres=0.01, min_dist=300)
    #peaks_x = peakutils.interpolate(wl[mask], img[:, 0], ind=peaks,width=50)
    peaks = np.array(np.round(peaks),dtype=np.int)
    print(peaks)
    plt.plot(wl[mask],img[:, 10])
    for xc in peaks:
        plt.axvline(x=wl[mask][xc])
    plt.tight_layout()
    plt.savefig(savedir +'fit_example' + ".png",dpi=600)
    plt.close()


    maxs = np.zeros(img.shape[1])
    peaks_x = np.zeros(img.shape[1],dtype=np.int)
    peakindexes = np.zeros(img.shape[1],dtype=np.int)

    amps = [0.005, 0.005,0.008, 0.02]
    x0s = [400, 560,600, 640]
    sigmas = np.repeat(20, len(amps))
    p0 = np.hstack((amps, x0s, sigmas, 0.0))
    last_p0 = p0
    n = int(len(amps))

    fitted_amps = []
    fitted_x0s = []
    fitted_sigmas = []

    for i in range(img.shape[1]):

        x = wl[mask]
        y = img[:,i]

        p0 = last_p0

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(x, y, linestyle='', marker='.')
        # y_fit = lorentzSum(x, *p0)
        # ax.plot(x, y_fit,color='black')
        # plt.tight_layout()
        # plt.savefig(savedir + 'fit_test' + ".png", dpi=300)
        # plt.close()

        upper = np.hstack((np.repeat(max(y), n), np.repeat(max(wl), n), np.repeat(1000, n),1))#+np.inf))
        lower = np.hstack((np.repeat(0, n), np.repeat(0, n), np.repeat(1, n),-1))#-np.inf))
        bounds = [lower,upper]

        try:
            popt, pcov = curve_fit(lorentzSum, x, y, p0,bounds=bounds,max_nfev=1000*len(x))
            perr = np.sqrt(np.diag(pcov))
            #print(popt)

            amps, x0s, sigmas, c = separate_parameters(popt)
            amps_err, x0s_err, sigmas_err, c_err = separate_parameters(perr)

            sorted = np.argsort(x0s)
            amps = amps[sorted]
            x0s = x0s[sorted]
            sigmas = sigmas[sorted]

            fitted_amps.append(amps)
            fitted_x0s.append(x0s)
            fitted_sigmas.append(sigmas)

            last_p0 = np.hstack((amps/2, x0s, sigmas, 0.0))
            popt = np.hstack((amps, x0s, sigmas, 0.0))

            indices = np.zeros(len(amps),dtype=np.int)
            for j in range(len(x0s)):
                indices[j] = (np.abs(x - x0s[j])).argmin()

            peakindexes[i] = x0s.max()
            maxs[i] = amps.max()

            if plot_fits:
                fig = plt.figure()
                ax = fig.add_subplot(111)

                for j in range(len(amps)):
                    ax.plot(x, lorentz(x, amps[j], x0s[j], sigmas[j]) )#+ c)

                ax.plot(x, y, linestyle='', marker='.')
                y_fit = lorentzSum(x, *popt)
                ax.plot(x, y_fit,color='black')

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
                plt.savefig(savedir + "plots/" + files[i][:-4] + ".png",dpi=600)
                #plt.show()
                plt.close()
        except:
            print("Error with "+ files[i])



    maxs = maxs/maxs[0]
    plt.plot(dt,maxs, color = "black", linewidth=1)
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$t [min]$')
    #plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir +'trace_max' + ".png",dpi=300)
    plt.close()

    maxs = np.zeros(img.shape[1])
    for i in range(img.shape[1]):
        maxs[i] = img[(np.argmin(np.abs(wl[mask]-750))),i]

    maxs = maxs/maxs[0]
    plt.plot(dt,maxs, color = "black", linewidth=1)
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r't [min]$')
    #plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir +'trace' + ".png",dpi=300)
    plt.close()

    maxs = wl[peakindexes]
    plt.plot(dt, maxs, color="black", linewidth=1)
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r't [min]$')
    #plt.ylim([595,615])
    # plt.legend(files)
    plt.tight_layout()
    plt.savefig(savedir + 'peak_wl' + ".png", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    for i in range(len(fitted_amps)):
        sca1 = ax.scatter(dt[i], fitted_amps[i][0]-np.mean(fitted_amps[0][0]), s=10, marker="x", color='C0', label=str(int(np.round(fitted_x0s[i][0],0)))+' nm')
        sca2 = ax.scatter(dt[i], fitted_amps[i][1]-np.mean(fitted_amps[0][1]), s=10, marker="o", color='C1', label=str(int(np.round(fitted_x0s[i][1],0)))+' nm')
        sca3 = ax.scatter(dt[i], fitted_amps[i][2]-np.mean(fitted_amps[0][2]), s=10, marker="D", color='C2', label=str(int(np.round(fitted_x0s[i][2],0)))+' nm')
        sca4 = ax.scatter(dt[i], fitted_amps[i][3]-np.mean(fitted_amps[0][3]), s=10, marker="+", color='C3', label=str(int(np.round(fitted_x0s[i][3],0)))+' nm')

    ax.legend(handles=[sca1, sca2, sca3, sca4], edgecolor='black', frameon=True)
    ax.set_ylabel("amplitude shift/ a.u.")
    ax.set_xlabel("time / min")
    plt.tight_layout()
    #plt.savefig(path + "all_amps.pdf", dpi=300)
    #plt.savefig(path + "all_amps.pgf")
    plt.savefig(savedir + "all_amps.png", dpi=600)
    plt.close()

    fig, ax = plt.subplots()
    for i in range(len(fitted_amps)):
        sca1 = ax.scatter(dt[i], fitted_x0s[i][0]-np.mean(fitted_x0s[0][0]), s=10, marker="x", color='C0', label=str(int(np.round(fitted_x0s[i][0],0)))+' nm')
        sca2 = ax.scatter(dt[i], fitted_x0s[i][1]-np.mean(fitted_x0s[0][1]), s=10, marker="o", color='C1', label=str(int(np.round(fitted_x0s[i][1],0)))+' nm')
        sca3 = ax.scatter(dt[i], fitted_x0s[i][2]-np.mean(fitted_x0s[0][2]), s=10, marker="D", color='C2', label=str(int(np.round(fitted_x0s[i][2],0)))+' nm')
        sca4 = ax.scatter(dt[i], fitted_x0s[i][3]-np.mean(fitted_x0s[0][3]), s=10, marker="+", color='C3', label=str(int(np.round(fitted_x0s[i][3],0)))+' nm')

    ax.legend(handles=[sca1, sca2, sca3, sca4], edgecolor='black', frameon=True)
    ax.set_ylabel("wavelength shift / nm")
    ax.set_xlabel("time / min")
    plt.tight_layout()
    #plt.savefig(path + "all_x0s.pdf", dpi=300)
    #plt.savefig(path + "all_x0s.pgf")
    plt.savefig(savedir + "all_x0s.png", dpi=600)
    plt.close()

    fig, ax = plt.subplots()
    for i in range(len(fitted_amps)):
        sca1 = ax.scatter(dt[i], fitted_sigmas[i][0]-np.mean(fitted_sigmas[0][0]), s=10, marker="x", color='C0', label=str(int(np.round(fitted_x0s[i][0],0)))+' nm')
        sca2 = ax.scatter(dt[i], fitted_sigmas[i][1]-np.mean(fitted_sigmas[0][1]), s=10, marker="o", color='C1', label=str(int(np.round(fitted_x0s[i][1],0)))+' nm')
        sca3 = ax.scatter(dt[i], fitted_sigmas[i][2]-np.mean(fitted_sigmas[0][2]), s=10, marker="D", color='C2', label=str(int(np.round(fitted_x0s[i][2],0)))+' nm')
        sca4 = ax.scatter(dt[i], fitted_sigmas[i][3]-np.mean(fitted_sigmas[0][3]), s=10, marker="+", color='C3', label=str(int(np.round(fitted_x0s[i][3],0)))+' nm')

    ax.legend(handles=[sca1, sca2, sca3, sca4], edgecolor='black', frameon=True)
    ax.set_ylabel("sigma shift / nn")
    ax.set_xlabel("time / min")
    plt.tight_layout()
    #plt.savefig(path + "all_x0s.pdf", dpi=300)
    #plt.savefig(path + "all_x0s.pgf")
    plt.savefig(savedir + "all_sigmas.png", dpi=600)
    plt.close()