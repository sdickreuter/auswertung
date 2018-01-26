__author__ = 'sei'

import os
import re
import sys

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter

from plotsettings import *

from gauss_detect import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import PIL
from matplotlib import ticker
import string
import itertools

path = '/home/sei/Spektren/artur/zeiss 100um vertical FIB/'

#samples = ['R55-110_D2.2','R55-110_D_2.1','Disk220-D2,2-1,3']
#samples = ['R55-110_D1.4_pPol_centred','R55-110_D1.6_pPol_centred','R55-110_D1.8_pPol_centred','R55-110_D2.0_pPol_centred','R55-110_D2.2_pPol_centred-1']
#samples = ['R55-110_D1.6_sPol_centred','R55-110_D1.8_sPol_centred','R55-110_D2.0_sPol_centred','R55-110_D2.2_sPol_centred','R55-110_D1.8_sPol_centred-1','R55-110_D1.8_sPol_centred-2','R55-110_D1.4_pPol_centred','R55-110_D1.6_pPol_centred','R55-110_D1.8_pPol_centred','R55-110_D2.0_pPol_centred','R55-110_D2.2_pPol_centred-1']


#samples = ['p52m_dif5_par']
#samples = ['p45m_did5_par','p45m_did6_par']
#samples = ['p41m_dif3_par','p41m_dif4_par','p41m_dif5_par']
#samples = ['ED30_1_test']

samples = []

dirs = [entry.path for entry in os.scandir(path) if entry.is_dir()]
for dir in dirs:
    if len(re.findall("old", dir)) == 0:
        if len(re.findall("Fluor", dir)) == 0:
            samples.append(dir)

if len(samples) < 1:
    samples = [path]

#for file in os.listdir(path):
#    if re.fullmatch(r"([A-Z]{1}[0-9]{1})(.csv)$", file) is not None:
#        files.append(file)
#listdir = os.listdir(path)


# dirs = [entry.path for entry in os.scandir(path+'Artur/All Scanned/') if entry.is_dir()]
# for dir in dirs:
#     #if re.fullmatch(r"(old){1}", dir) is None:
#     samples.append(dir)

maxwl = 950
minwl = 450



#raise RuntimeError()

def get_letters(size):
    def iter_all_ascii():
        size = 1
        while True:
            for s in itertools.product(string.ascii_uppercase, repeat=size):
                yield "".join(s)
            size += 1

    letters = np.array([])
    for s in itertools.islice(iter_all_ascii(), size):
        letters = np.append(letters, s)
    return letters


def get_numbers(size):
    def iter_all_numbers():
        size = 1
        while True:
            for s in itertools.product(string.digits[0:10], repeat=size):
                yield "".join(s)
            size += 1

    numbers = np.array([])
    for s in itertools.islice(iter_all_numbers(), size):
        numbers = np.append(numbers, s)
    return numbers


def make_plots(sample):
    print(sample)

    #savedir = path + sample + '/'
    savedir = sample + '/'

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
    wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

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
    plt.plot(wl[mask], bg[mask])
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/bg.pdf", dpi=300)
    plt.close()
    plt.plot(wl[mask], dark[mask])
    plt.xlim((minwl, maxwl))
    plt.savefig(savedir + "overview/dark.pdf", dpi=300)
    plt.close()

    # files = []
    # for file in os.listdir(savedir):
    #     if re.fullmatch(r"([A-Z]{1}[0-9]{1,3}?)(.csv)$", file) is not None:
    #         files.append(file)
    # files = np.sort(files)

    is_grid_data = True
    files = []
    listdir = os.listdir(savedir)
    letters = get_letters(1000)
    numbers = get_numbers(1000)
    for l in letters:
        for n in numbers:
            if l+n+'.csv' in listdir:
                files.append(l+n+'.csv')

    if len(files) < 1:
        is_grid_data = False
        for file in listdir:
            if re.search(r"(.csv)$", file) is not None:
                if file != 'background.csv' and file != 'lamp.csv' and file != 'dark.csv' and file != 'normal.csv':
                    files.append(file)

    print(len(files))

    # file = files[10]
    # wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
    # if not is_extinction:
    #     counts = (counts - bg) / (lamp - dark)
    # else:
    #     counts = 1 - (counts - dark) / (lamp - dark)
    # plt.plot(wl,counts)
    # plt.show()

    img = np.zeros((len(files),len(lamp)))
    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        counts = (counts - bg) / (lamp - dark)

        counts[np.where(counts == np.inf)] = 0.0
        #filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        img[i,:] = counts


    if is_grid_data:
        newfig(0.9)
        plt.imshow(img, aspect='auto',cmap=plt.get_cmap("viridis"),extent=[wl.min(),wl.max(),0,len(files)],norm=LogNorm())
        plt.ylabel(r'$number of measurement$')
        plt.xlabel(r'$\lambda\, /\, nm$')
        plt.savefig(savedir + "overview/image_log.pdf")
        plt.savefig(savedir + "overview/image_log.png", dpi=400)
        #plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
        plt.close()

        img = np.zeros((len(files),len(lamp)))
        for i in range(len(files)):
            file = files[i]
            wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
            counts = (counts - bg) / (lamp - dark)

            counts[np.where(counts == np.inf)] = 0.0
            #filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
            img[i,:] = counts


        newfig(0.9)
        plt.imshow(img, aspect='auto',cmap=plt.get_cmap("viridis"),extent=[wl.min(),wl.max(),0,len(files)])
        plt.ylabel(r'$number of measurement$')
        plt.xlabel(r'$\lambda\, /\, nm$')
        plt.savefig(savedir + "overview/image.pdf")
        plt.savefig(savedir + "overview/image.png", dpi=400)
        #plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
        plt.close()




    # for i in range(len(files)):
    #     file = files[i]
    #     wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
    #     #wl = wl[mask]
    #     if not is_extinction:
    #         counts = (counts - bg) / (lamp - dark)
    #     else:
    #         counts = 1 - (counts - dark) / (lamp - dark)
    #
    #     filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    #
    #     # counts = (counts-dark)/(lamp-dark)
    #     #counts = counts[mask]
    #     # plt.plot(wl, counts,color=colors[i],linewidth=0.6)
    #     plt.plot(wl[mask], filtered[mask], linewidth=0.6)
    #
    # plt.xlim((minwl, maxwl))
    # plt.ylabel(r'$I_{df} [a.u.]$')
    # plt.xlabel(r'$\lambda [nm]$')
    # plt.tight_layout()
    # plt.savefig(savedir + 'Overview' + ".pdf", dpi=200)
    # plt.close()


    for i in range(len(files)):
        print('Plotting '+savedir+file)
        if is_grid_data:
            savename = str(i).zfill(6)
        else:
            savename = file

        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        counts = (counts - bg) / (lamp - dark)

        counts[np.where(counts == np.inf)] = 0.0
        filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        newfig(0.9)
        plt.plot(wl[mask], counts[mask], color="grey", linewidth=0.5)
        plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.5)
        plt.ylabel(r'$I_{df}\, /\, a.u.$')
        plt.xlabel(r'$\lambda\, /\, nm$')
        plt.xlim((minwl, maxwl))
        plt.ylim([np.min(filtered[mask]), np.max(filtered[mask]) * 1.2])
        #plt.ylim([np.min(img), np.max(img) * 1.1])
        plt.tight_layout()
        plt.savefig(savedir + "plots/" + savename + ".png", dpi=400)
        #plt.savefig(savedir + "plots/" + file[:-4] + ".pdf", dpi=300)
        #plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
        plt.close()

        f = open(savedir + "specs/" + savename + ".csv", 'w')
        f.write("wavelength,intensity" + "\r\n")
        for z in range(len(counts)):
            f.write(str(wl[z]) + "," + str(counts[z]) + "\r\n")
        f.close()



for sample in samples:
    try:
        make_plots(sample)
    except FileNotFoundError:
        print(sample+' has no background/dark/lamp spectra')

#
# if len(sys.argv) == 2:
#     dir = sys.argv[1]
# else:
#     RuntimeError("Too much/less arguments")
#
# make_plots(dir)
