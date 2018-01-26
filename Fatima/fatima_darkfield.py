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
from matplotlib.patches import Polygon
import pandas as pd

path = '/home/sei/Spektren/fatima/'

samples = ['ED30_Batch3(1)','ED30_Batch3(2)','ED90_Batch3(1)','ED90_Batch3(2)']


# grid dimensions
nx = 5
ny = 5
maxwl = 800
minwl = 400

letters = [chr(c) for c in range(65, 91)]

wl, lamp = np.loadtxt(open(path + samples[0] + "/lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

check_wavelengths = [630,637,640]
mean_values = np.zeros((len(samples),len(check_wavelengths)))
err_values = np.zeros((len(samples),len(check_wavelengths)))

mean_specs = np.zeros((len(samples),len(wl)))
err_specs = np.zeros((len(samples),len(wl)))



for s,sample in enumerate(samples):
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
    bg = dark

    #wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

    mask = (wl >= minwl) & (wl <= maxwl)

    sns.set_style("ticks", {'axes.linewidth': 1.0,
                            'xtick.direction': 'in',
                            'ytick.direction': 'in',
                            'xtick.major.size': 3,
                            'xtick.minor.size': 1.5,
                            'ytick.major.size': 3,
                            'ytick.minor.size': 1.5
                            })

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
    # plt.close()
    #
    files = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([A-Z]{1}[0-9]{1})(.csv)$", file) is not None:
            files.append(file)

    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        #wl = wl[mask]
        counts = (counts - bg) / (lamp - dark)
        filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        # counts = (counts-dark)/(lamp-dark)
        #counts = counts[mask]
        # plt.plot(wl, counts,color=colors[i],linewidth=0.6)
        plt.plot(wl[mask], filtered[mask], linewidth=0.6)

    plt.xlim((minwl, maxwl))
    plt.ylabel(r'$I_{df}\, /\, a.u.$')
    plt.xlabel(r'$\lambda\, /\, nm$')
    plt.tight_layout()
    plt.savefig(savedir + 'Overview' + ".pdf", dpi=200)
    plt.close()

    specs = np.zeros((len(files),len(counts)))
    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        counts = (counts - bg) / (lamp - dark)
        filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        specs[i,:] = filtered

    mean_spec = np.zeros(specs.shape[1])
    err_spec = np.zeros(specs.shape[1])
    for i in range(specs.shape[1]):
        mean_spec[i] = np.mean(specs[:,i])
        err_spec[i] = np.std(specs[:, i])

    poly = np.array((wl,mean_spec+err_spec))
    poly = np.hstack((poly,np.fliplr(np.array((wl, mean_spec - err_spec)))))
    poly = poly.T
    print(poly.shape)
    #plt.Polygon(poly)

    # fig, ax = newfig(0.9)
    # ax.add_patch(Polygon(poly, closed=True,fill=True,facecolor='0.85'))
    # plt.plot(wl, mean_spec, linewidth=0.8)
    # plt.xlim((minwl, maxwl))
    # plt.ylim((0,np.max(mean_spec+err_spec)*1.01))
    # plt.ylabel(r'$I_{df} [a.u.]$')
    # plt.xlabel(r'$\lambda [nm]$')
    # plt.tight_layout()
    # plt.savefig(savedir + 'Mean' + ".pdf", dpi=200)
    # plt.close()

    for i,wavelength in enumerate(check_wavelengths):
        index = (np.abs(wl - wavelength)).argmin()
        mean_values[s,i] = mean_spec[index]
        err_values[s,i] = err_spec[index]

    mean_specs[s,:] = mean_spec
    err_specs[s, :] = err_spec


#err_specs /= mean_specs.max()
#mean_specs /= mean_specs.max()


fig, ax = newfig(0.9)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

print(samples)
legend = ["A: 30 min","B: 30 min","C: 90 min","D: 90 min"]
print(legend)

for s in range(len(samples)):
    mean_spec = mean_specs[s,:]#+s/3
    err_spec = err_specs[s,:]
    poly = np.array((wl,mean_spec+err_spec))
    poly = np.hstack((poly,np.fliplr(np.array((wl, mean_spec - err_spec)))))
    poly = poly.T
    ax.add_patch(Polygon(poly, closed=True,fill=True,alpha = 0.3,facecolor=colors[s]))
    plt.plot(wl, mean_spec, linewidth=0.8)

plt.xlim((minwl, maxwl))
plt.ylim((0, np.max(mean_spec + err_spec) * 1.0))
plt.ylabel(r'$I_{df}\, /\, a.u.$')
plt.xlabel(r'$\lambda\, /\, nm$')
plt.legend(legend)
plt.tight_layout()
plt.savefig(path + 'Mean' + ".pdf", dpi=200)
plt.close()



# rem = pd.read_csv('/home/sei/REM/Fatima2/' + 'coverage.csv')
# coverage = rem['coverage']
#
# fig, ax = newfig(0.9)
#
# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#
# for s in range(len(samples)):
#     mean_spec = mean_specs[s,:]/coverage[s]
#     err_spec = err_specs[s,:]/coverage[s]
#     poly = np.array((wl,mean_spec+err_spec))
#     poly = np.hstack((poly,np.fliplr(np.array((wl, mean_spec - err_spec)))))
#     poly = poly.T
#     ax.add_patch(Polygon(poly, closed=True,fill=True,alpha = 0.3,facecolor=colors[s]))
#     plt.plot(wl, mean_spec, linewidth=0.8)
#
# plt.xlim((minwl, maxwl))
# plt.ylim((0, np.max(mean_spec + err_spec) * 1.0))
# plt.ylabel(r'$I_{df}/\text{fill-factor} [a.u.]$')
# plt.xlabel(r'$\lambda [nm]$')
# plt.legend(legend)
# plt.tight_layout()
# plt.savefig(path + 'Mean_by_coverage' + ".pdf", dpi=200)
# plt.close()



f = open(path + "values.csv", 'w')

f.write('label,')

for wavelength in check_wavelengths:
    f.write('mean'+str(wavelength) + ',')

for wavelength in check_wavelengths:
    f.write('err'+str(wavelength)+',')

f.write("\r\n")

for s in range(len(samples)):
    f.write(samples[s] + ',')

    for i in range(len(check_wavelengths)):
        f.write(str(mean_values[s,i]) + ',')

    for i in range(len(check_wavelengths)):
        f.write(str(err_values[s,i]) + ',')

    f.write("\r\n")
f.close()


        # for i in range(len(files)):
    #     file = files[i]
    #     wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
    #     counts = (counts - bg) / (lamp - dark)
    #
    #     counts[np.where(counts == np.inf)] = 0.0
    #     filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    #     newfig(0.9)
    #     plt.plot(wl[mask], counts[mask], color="grey", linewidth=0.5)
    #     plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.5)
    #     plt.ylabel(r'$I_{df}\, /\, a.u.$')
    #     plt.xlabel(r'$\lambda\, /\, nm$')
    #     plt.xlim((minwl, maxwl))
    #     plt.ylim([np.min(filtered[mask]), np.max(filtered[mask]) * 1.2])
    #     plt.tight_layout()
    #     plt.savefig(savedir + "plots/" + file[:-4] + ".pdf", dpi=300)
    #     #plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
    #     plt.close()
    #
    #     f = open(savedir + "specs/" + file[:-4] + ".csv", 'w')
    #     f.write("wavelength,intensity" + "\r\n")
    #     for z in range(len(counts)):
    #         f.write(str(wl[z]) + "," + str(counts[z]) + "\r\n")
    #     f.close()
    #
