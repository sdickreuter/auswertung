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

try:
    import cPickle as pickle
except ImportError:
    import pickle

path = '/home/sei/Spektren/pranoti/'

files = []
for file in os.listdir(path):
    if file.endswith("line.pkl"):
        files.append(file)

print(files)

seconds = []
for file in files:
    search = re.search(r"([0-9]{0,2}[.]{0,1}[0-9]{1,2})(s)", file)
    if search is not None:
        seconds.append(search.group(0)[:-1])
    #elif re.search(r"(graphene)", file) is not None:

print(seconds)

seconds = np.array(seconds, dtype=np.float)
sorted = np.argsort(seconds)
seconds = seconds[sorted]
files = np.array(files)
files = files[sorted]

legend = []
for s in seconds:
    legend.append(str(s)+' s')

fig = plt.figure()
ax = plt.subplot(111)

for i in range(len(files)):
    with open(path+files[i], 'rb') as fp:
        x, transmittance = pickle.load(fp)
        buf = np.argmin(transmittance[:-int(transmittance.shape[0]/2)])
        x = x - x[buf]+10
        ax.plot(x, transmittance,label=legend[i])

ax.set_xlabel(r'$x\, /\, \mu m$')
ax.set_ylabel(r'$T^{rel}_{400-700\,nm}$')
ax.set_xlim((0,50))
ax.set_ylim((0, 1.35))
#plt.legend(legend)
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3)#, fancybox=True, shadow=True)
#ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.tight_layout()
plt.savefig(path +  "line_overview.png", dpi=1200, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()


T = []
T_std = []
for f in files:
    with open(path+f, 'rb') as fp:
        x, transmittance = pickle.load(fp)
        buf = np.argmin(transmittance[:-int(transmittance.shape[0]/2)])
        x = x - x[buf]+10
        T.append( np.mean(transmittance[ (x>15) & (x<35)] ))
        T_std.append(np.std(transmittance[(x > 15) & (x < 35)]))


fig, ax1 = plt.subplots()

(_, caps, _) = ax1.errorbar(seconds, T, yerr=T_std, fmt='.', elinewidth=0.5, markersize=6, capsize=4,color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)

ax1.set_xlabel(r'PEDOT time $/\, s$')
ax1.set_ylabel(r'$T^{rel}_{400-700\,nm}$')
plt.tight_layout()
plt.savefig(path + "integrated_line.png", dpi=1200)
plt.close()

f = open(path + "transmittance.csv", 'w')
f.write("seconds,T,T_std" + "\r\n")
for i in range(len(seconds)):
    f.write(str(seconds[i]) + "," + str(T[i]) + "," + str(T_std[i]) +  "\r\n")

f.close()

