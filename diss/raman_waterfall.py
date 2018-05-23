
import numpy as np
from plotsettings import *
import matplotlib.pyplot as plt
import os
import re
import PIL
import matplotlib.gridspec as gridspec
import matplotlib.mlab as mlab
import pandas as pd
from scipy import signal
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import stats
from scipy.optimize import minimize, basinhopping
from adjustText import adjust_text
import pickle
from matplotlib.mlab import griddata

path = '/home/sei/Raman/2C1/'
sample = '2C1_75hept_B2'
#sample = '2C1_100hex_C2'
#sample = '2C1_150hept_B2'
#sample = '2C1_150hex_B3'
#sample = '2C1_200hex_A2'


savedir = path + sample + '_plots/'
loaddir = path + sample + '/'

substrate = '2C1'
sem_dir = '/home/sei/Auswertung/'+substrate+'/'

sem = pd.read_csv(sem_dir+sample+"/particles_SEM.csv", sep=',')
sem["area"] *= 6.201172**2 # [nm²]
sem_area = sem["area"]
sem_ids = sem["id"]

with open(path + sample+'_positions_redo.pkl', 'rb') as fp:
    x, y, pos_files = pickle.load(fp)

x-=x.min()
y-=y.min()

with open(path + sample+ '_grid.pkl', 'rb') as fp:
    gridx, gridy, ids = pickle.load(fp)

gridy+=0.2

files = []
new_ids = []
for i in range(len(gridx)):
    dists = np.ones(len(x))*1e12
    for j in range(len(x)):
        dists[j] = np.sqrt( (x[j]-gridx[i])**2 + (y[j]-gridy[i])**2)
    ind = np.argmin(dists)
    files.append(pos_files[ind])

files = np.array(files)

plot_ids = []
plot_files = []
plot_areas = []
for r in range(len(ids)):
    for s in range(len(sem_ids)):
        if ids[r] == sem_ids[s]:
            plot_ids.append(ids[r])
            plot_files.append(files[r])
            plot_areas.append(sem_area[s])

plot_ids = np.array(plot_ids)
plot_files = np.array(plot_files)
plot_areas = np.array(plot_areas)


int = np.zeros(len(pos_files))
for i in range(len(pos_files)):
    wl, counts = np.loadtxt(open(loaddir+pos_files[i], "rb"), delimiter="\t", skiprows=1, unpack=True)
    int[i] = counts[(wl > 1580) & (wl < 1590)].max()

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
# grid the data.
zi = griddata(x, y, int, xi, yi, interp='linear')
# contour the gridded data, plotting dots at the nonuniform data points.
#CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = plt.contourf(xi, yi, zi, 15,
                  vmax=int.max(), vmin=int.min())
plt.plot(gridx,gridy,'rx')
plt.show()
plt.close()


sorted = np.argsort(plot_areas)
plot_areas = plot_areas[sorted]
plot_files = plot_files[sorted]
plot_ids = plot_ids[sorted]

yticks = []
labels_waterfall = []
max_int = 0

#y_pos = dist/dist.max()
y_pos = np.linspace(0,1,len(plot_areas))*10
print(y_pos)

maximum = 0
for f in plot_files:
    wl, counts = np.loadtxt(open(loaddir+f, "rb"), delimiter="\t", skiprows=1, unpack=True)
    maximum = np.max([maximum,counts.max()])

cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 1, len(plot_files))]

fig, ax = newfig(0.9,3.0)


for i, f in zip(range(len(plot_files)),plot_files):

    wl, counts = wl, counts = np.loadtxt(open(loaddir+f, "rb"), delimiter="\t", skiprows=1, unpack=True)
    counts/=maximum

    ax.plot(wl,counts+y_pos[i], linewidth=1.0,color = colors[i])

    yticks.append(y_pos[i])

    max_int = np.max([max_int,counts.max()+y_pos[i]])

    labels_waterfall.append(plot_ids[i]+': '+str(round(plot_areas[i]/1000,1))+'k')


yticks.append(y_pos[i]+np.diff(y_pos)[0])
labels_waterfall.append('Fläche / nm$^2$')


#print(peaks)
#print(peaks_err)
ax.set_ylabel(r'$I_{df}\, /\, a.u.$')
ax.set_xlabel(r'$\lambda\, /\, nm$')
#ax.set_ylim([0, (len(pics)+1)*dist_fac*1.1])
ax.set_ylim([-0.1, max_int*1.01])

ax.tick_params(axis='y', which='both',left='off',right='off', labelleft='off', labelright='on')
ax.set_yticks(yticks)
ax.set_yticklabels(labels_waterfall)

plt.tight_layout()
#plt.show()
plt.savefig(savedir + "waterfall.pdf", dpi= 400)
plt.savefig(savedir + "waterfall.pgf")
plt.savefig(savedir + "waterfall.png", dpi= 400)
plt.close()

