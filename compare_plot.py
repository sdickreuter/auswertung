import numpy as np

from plotsettings import *

import os
import re
import scipy

path = '/home/sei/Spektren/test2/'



maxwl = 1010
minwl = 420

wl, lamp = np.loadtxt(open(path + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, dark = np.loadtxt(open(path + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, bg = np.loadtxt(open(path + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

mask = (wl >= minwl) & (wl <= maxwl)


#files = []
#for file in os.listdir(path):
#    if re.fullmatch(r"([0-9]{5})(.csv)$", file) is not None:
#        files.append(file)

#files1 = ['lockin_export.csv','lockin_export_p45m.csv']

files1 = []
for file in os.listdir(path):
   if re.match(r"(lockin_){1}(\w)+(.csv)$", file) is not None:
       files1.append(file)

print(files1)

files2 = []
for file in os.listdir(path):
   if re.match(r"(mean_){1}(\w)+(.csv)$", file) is not None:
       files2.append(file)

print(files2)



for i in range(len(files1)):
    file1 = files1[i]
    file2 = files2[i]

    d = np.genfromtxt(open(path + file1, "rb"), delimiter=",", skip_header=1)
    wl = d[:,0]
    counts1 = d[:,1]
    #counts = (counts-bg)/(lamp-dark)
    #counts1 = (counts1-dark)/(lamp-dark)

    int = np.zeros(wl.shape[0])
    for j in range(wl.shape[0]-1):
        int[j+1] = scipy.integrate.simps(counts1[:j+1],wl[:j+1])


    wl = wl[mask]
    int = int[mask]
    int = int / np.max(np.abs(int))

    counts1 = counts1[mask]
    counts1 = counts1/np.max(np.abs(counts1))


    d = np.genfromtxt(open(path + file2, "rb"), delimiter=",", skip_header=1)
    wl = d[:,0]
    counts2 = d[:,1]
    wl = wl[mask]
    #counts = (counts-bg)/(lamp-dark)
    #counts2 = (counts2-dark)/(lamp-dark)
    counts2 = counts2[mask]
    counts2 = counts2/np.max(counts2)


    fig = newfig(0.9)
    plt.plot(wl, counts1)
    plt.plot(wl, counts2)
    plt.plot(wl, int)
    plt.xlim((minwl,maxwl))
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$\lambda [nm]$')
    plt.tight_layout()
    # plt.plot(wl, counts)
    plt.savefig(path+file1[:-4] + ".png",dpi=200)
    plt.close()


