_author__ = 'sei'

import os

import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import scipy.optimize as opt
from plotsettings import *
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
import pandas as pd
import re
import pickle

#path = '/home/sei/Raman/2C2/'
#sample = '2C2_75hept_B2'
#sample = '2C2_150hex_C2'
#sample = '2C2_150tri_A1'
#sample = '2C2_200hex_B1'
#sample = '2C2_200tri_A3'

path = '/home/sei/Raman/2C1/'
sample = '2C1_75hept_B2'
#sample = '2C1_100hex_C2'
#sample = '2C1_150hept_B2'
#sample = '2C1_150hex_B3'
#sample = '2C1_200hex_A2'


savedir = path + sample + '_singleplots/'
loaddir = path + sample + '/'

try:
    os.mkdir(savedir)
except:
    pass

files = []
n = []
for file in os.listdir(loaddir):
    if re.search("_{1}[0123456789]+_{1}", file) is not None:
        m = re.search("_{1}[0123456789]+_{1}", file)
        n.append(int(m.group(0)[1:-1]))
        files.append(file)

n = np.array(n)
files = np.array(files)

ordered = np.argsort(n)
n = n[ordered]
files = files[ordered]

x = []
y = []
for f in files:
    reg = re.search("X{1}_{1}-?[0123456789]+.{1}[0123456789]+",f)
    x.append(float(reg.group(0)[2:]))

    reg = re.search("Y{1}_{1}-?[0123456789]+.{1}[0123456789]+", f)
    y.append(float(reg.group(0)[2:]))

x = np.array(x)
y = np.array(y)

f = files[100]

for f in files:

    fig, ax = newfig(0.9)

    wl, counts = np.loadtxt(open(loaddir+f, "rb"), delimiter="\t", skiprows=1, unpack=True)

    ax.plot(wl, counts)

    ax.set_ylabel(r'$I_{df}\, /\, a.u.$')
    ax.set_xlabel(r'$\lambda\, /\, nm$')
    plt.tight_layout()
    #plt.show()
    plt.savefig(savedir + f +".pgf")
    plt.savefig(savedir + f +".png", dpi= 400)
    plt.close()
