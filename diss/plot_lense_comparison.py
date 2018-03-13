import numpy as np

from plotsettings import *

import matplotlib as mpl

import os
import re
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

path2 = '/home/sei/Spektren/A1_newlense_newsetup_100um/'
path1 = '/home/sei/Spektren/A1_oldlense_newsetup_100um/'


maxwl = 900
minwl = 600

wl, lamp1 = np.loadtxt(open(path1 + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, dark1 = np.loadtxt(open(path1 + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, bg1 = np.loadtxt(open(path1 + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

wl, lamp2 = np.loadtxt(open(path2 + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, dark2 = np.loadtxt(open(path2 + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, bg2 = np.loadtxt(open(path2 + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)


mask = (wl >= minwl) & (wl <= maxwl)

files1 = []
z1 = []
for file in os.listdir(path1):
    if re.fullmatch(r"([0-9]{1,2})(.csv)$", file) is not None:
        files1.append(file)
        m = re.match(r"([0-9]{1,2})", file)
        z1.append(int(m.group(0)))

z1 = np.array(z1)
files1 = np.array(files1)
sorted_ind = np.argsort(z1)
z1 = z1[sorted_ind]
files1 = files1[sorted_ind]


files2 = []
z2 = []
for file in os.listdir(path2):
    if re.fullmatch(r"([0-9]{1,2})(.csv)$", file) is not None:
        files2.append(file)
        m = re.match(r"([0-9]{1,2})", file)
        z2.append(int(m.group(0)))

z2 = np.array(z2)
files2 = np.array(files2)
sorted_ind = np.argsort(z2)
z2 = z2[sorted_ind]
files2 = files2[sorted_ind]

#files = np.flip(files,0)
print(z1)
print(files1)


img1 = np.zeros((lamp1[mask].shape[0],len(files1)))

for i in range(len(files1)):
    file = files1[i]
    wl, counts = np.loadtxt(open(path1+file, "rb"), delimiter=",", skiprows=16, unpack=True)
    wl = wl[mask]
    counts = (counts-bg1)/(lamp1-dark1)
    counts = savgol_filter(counts,51,0)
    counts = counts[mask]
    img1[:,i] = counts


img2 = np.zeros((lamp2[mask].shape[0],len(files2)))

for i in range(len(files2)):
    file = files2[i]
    wl, counts = np.loadtxt(open(path2+file, "rb"), delimiter=",", skiprows=16, unpack=True)
    wl = wl[mask]
    counts = (counts-bg2)/(lamp2-dark2)
    counts = savgol_filter(counts,51,0)
    counts = counts[mask]
    img2[:,i] = counts


# sizes =figsize(1.3)
# fig, axes = plt.subplots(ncols=2, figsize=(sizes[0], sizes[1]),sharey=True)
# ax1, ax2 = axes
#
# cmap = plt.get_cmap('plasma')#sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
# colors = cmap(np.linspace(0.1, 1, len(files1)))
#
#
# for i in range(img1.shape[1]):
#     ax1.plot(wl, img1[:, i] + i * 0.0003, linewidth=1, color=colors[i],zorder=img1.shape[1]-i)
#
# for i in range(img2.shape[1]):
#     ax2.plot(wl, img2[:, i] + i * 0.0003, linewidth=1, color=colors[i],zorder=img2.shape[1]-i)
#
# m = plt.cm.ScalarMappable(cmap=cmap)
# m.set_array(np.linspace(0,1,len(files1)))
# cb = plt.colorbar(m)
# #cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
# #cb = plt.colorbar(m, cax=cax, **kw)
#
# #cbaxes = fig.add_axes([0.1, 0.1, 0.8, 0.1]) # setup colorbar axes.
# # put the colorbar on new axes
# #cb = fig.colorbar(m,cax=cbaxes,orientation='vertical')
# #cb = fig.colorbar(m, cax=cbaxes)
#
# #tick_locator = mpl.ticker.MaxNLocator(nbins=5)
# #cb.locator = tick_locator
# #cb.update_ticks()
# #cb.ax.tick_params(axis='y', direction='out')
# #divider = make_axes_locatable(axes)
# #cax = divider.append_axes("right", size="5%", pad=0.2)
#
# #cb = plt.colorbar(m, cax=cax)
#
#
# cb.set_label(r'$z [\mu m]$')
#
# plt.xlim((minwl, maxwl))
# plt.ylabel(r'$I_{df} [a.u.]$')
# plt.xlabel(r'$\lambda [nm]$')
# #plt.legend(files)
# #plt.tight_layout()
# plt.savefig('Lenses' + ".png",dpi=300)
# plt.savefig('Lenses' + ".pgf")
# plt.close()

sizes =figsize(1.3)
fig = plt.figure(figsize=(sizes[0], sizes[1]))

gs=mpl.gridspec.GridSpec(1,3, width_ratios=[4,4,0.2])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])

cmap = plt.get_cmap('plasma')#sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
colors = cmap(np.linspace(0.1, 1, len(files1)))

for i in range(img1.shape[1]):
    ax1.plot(wl, img1[:, i] + i * 0.0003, linewidth=1, color=colors[i],zorder=img1.shape[1]-i)

for i in range(img2.shape[1]):
    ax2.plot(wl, img2[:, i] + i * 0.0003, linewidth=1, color=colors[i],zorder=img2.shape[1]-i)

m = plt.cm.ScalarMappable(cmap=cmap)
m.set_array(z1-10)
plt.setp(ax2.get_yticklabels(), visible=False)
cb = plt.colorbar(m, cax=ax3)
cb.set_label(r'$z\, /\, \mu m$')

ax1.set_title("a) Achromat")
ax1.set_xlim((minwl, maxwl))
ax1.set_xlabel(r'$\lambda\, /\, nm$')
ax1.set_ylabel(r'$I_{df}\, /\, a.u.$')

ax2.set_title("b) Apochromat")
ax2.set_xlim((minwl, maxwl))
ax2.set_xlabel(r'$\lambda\, /\, nm$')


#plt.xlim((minwl, maxwl))
#plt.ylabel(r'$I_{df} [a.u.]$')
#plt.xlabel(r'$\lambda [nm]$')
#plt.legend(files)
plt.tight_layout()
plt.savefig('Lenses' + ".png",dpi=300)
plt.savefig('Lenses' + ".pgf")
plt.close()
