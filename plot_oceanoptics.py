import numpy as np

from plotsettings import *

import os
import re

import seaborn as sns

sns.set_context("poster", rc= {  "xtick.major.width":0.5,
                            "ytick.major.width":0.5,
                            "xtick.minor.width":0.5,
                            "ytick.minor.width":0.5})



path = '/home/sei/Spektren/PosterPlot/'


maxwl = 800
minwl = 460

wl, counts = np.loadtxt(open(path + "E1_corr.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

mask = (wl >= minwl) & (wl <= maxwl)

wl = wl[mask]
counts = counts[mask]
counts /= counts.max()


fig = newfig(1.1)
plt.plot(wl, counts)
plt.xlim((minwl,maxwl))
plt.ylabel(r'\boldmath$Intensit"at \; / \; a.u.$')
plt.xlabel(r'\boldmath$Wellenl"ange \; / \; nm$')
plt.tight_layout()
# plt.plot(wl, counts)
plt.savefig(path+"spec.png",dpi=600)
plt.close()
