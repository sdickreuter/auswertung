import numpy as np

# from plotsettings import *

import os
import re

import matplotlib
#import seaborn as sns
matplotlib.use('QT4Agg')

import hyperspy.api as hs
import matplotlib.pyplot as plt


#path = '/home/sei/Spektren/p57m_did6_01_map/'
path = '/home/sei/Spektren/SD504_F41A_Epobenlinks/'
#path = '/home/sei/Spektren/p57m_trie1_xline/'



maxwl = 750#970
minwl = 550#430

wl, lamp = np.loadtxt(open(path + "lamp.csv", "rb"), delimiter=",", skiprows=8, unpack=True)
wl, dark = np.loadtxt(open(path + "dark.csv", "rb"), delimiter=",", skiprows=8, unpack=True)
wl, bg = np.loadtxt(open(path + "background.csv", "rb"), delimiter=",", skiprows=8, unpack=True)

mask = (wl >= minwl) & (wl <= maxwl)

#plt.plot(wl, lamp - dark)
#plt.savefig(path + "lamp.png")
# plt.close()
# plt.plot(wl[mask], bg[mask])
# plt.savefig(path + "plots/bg.png")
# plt.close()
# plt.plot(wl[mask], dark[mask])
# plt.savefig(path + "plots/dark.png")
# plt.close()

files = []
for file in os.listdir(path):
    if re.fullmatch(r"([0-9]{5})(.csv)$", file) is not None:
        files.append(file)

files.sort()
#files = glob.glob(path+'/*.csv')
#files.sort(key=os.path.getmtime)

# try:
#     os.mkdir(path + "plots/")
# except:
#     pass

#print(files)

#n = int(np.sqrt(len(files)))
n = len(files)
d = np.zeros((len(files),len(wl[mask])))

for i in range(len(files)):
    file = files[i]
    wl, counts = np.loadtxt(open(path+file, "rb"), delimiter=",", skiprows=12, unpack=True)
    counts = (counts-bg)/(lamp-dark)
    #counts = (counts-dark)/(lamp-dark)
    counts = counts[mask]
    d[i,:] = counts

#d = d.reshape((n,n,len(wl[mask])))

s = hs.signals.Signal1D(d)
#s.axes_manager[0].name = "X"
#s.axes_manager[1].name = "X"
#s.axes_manager[2].name = "Scattering"

s.plot()
plt.show()

ax = hs.plot.plot_spectra(s, style="heatmap")
ax.images[0].set_cmap(matplotlib.cm.jet)
plt.show()



# for file in files:
#     wl, counts = np.loadtxt(open(path+file, "rb"), delimiter=",", skiprows=16, unpack=True)
#     wl = wl[mask]
#     counts = (counts-bg)/(lamp-dark)
#     #counts = (counts-dark)/(lamp-dark)
#     counts = counts[mask]
#
#     fig = newfig(0.9)
#     plt.plot(wl, counts)
#     plt.xlim((minwl,maxwl))
#     plt.ylabel(r'$I_{df} [a.u.]$')
#     plt.xlabel(r'$\lambda [nm]$')
#     plt.tight_layout()
#     # plt.plot(wl, counts)
#     plt.savefig(path+file[:-4] + ".png",dpi=200)
#     plt.close()

# fig = newfig(0.9)
# cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
# colors = cmap(np.linspace(0.1,1,len(files)))
#
# meanspec = np.zeros(lamp.shape)
#
# for i in range(len(files)):
#     file = files[i]
#     wl, counts = np.loadtxt(open(path+file, "rb"), delimiter=",", skiprows=16, unpack=True)
#     wl = wl[mask]
#     counts = (counts-bg)/(lamp-dark)
#     #counts = (counts-dark)/(lamp-dark)
#     meanspec += counts
#     counts = counts[mask]
#     #plt.plot(wl, counts,color=colors[i],linewidth=0.6)
#     plt.plot(wl, counts,linewidth=0.6)
#
# meanspec /= len(files)
# plt.plot(wl, meanspec[mask], color = "black", linewidth=1)
# plt.xlim((minwl, maxwl))
# plt.ylabel(r'$I_{df} [a.u.]$')
# plt.xlabel(r'$\lambda [nm]$')
# plt.tight_layout()
# plt.savefig(path +'Overview' + ".png",dpi=200)
# plt.close()