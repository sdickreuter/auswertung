
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

#path = '/home/sei/Spektren/2C1/'
#sample = '2C1_75hept_B2'
#sample = '2C1_100hex_C2'
#sample = '2C1_150hept_B2'
#sample = '2C1_150hex_B3'
#sample = '2C1_200hex_A2'




path = '/home/sei/Spektren/8/'
samples = ['8_C3_horzpol']
sample_names = ['C3']

savedir = path

files = []
ids = []
plot_ids = []
plot_files = []
plot_areas = []

for i in range(len(samples)):
    sample = samples[i]
    name = sample_names[i]


    loaddir = path + sample + '/specs/'

    maxwl = 900
    minwl = 400

    substrate = '2C1'
    sem_dir = '/home/sei/REM/8/'
    sem = pd.read_csv(sem_dir+sample[:4]+"_particles_SEM.csv", sep=',')
    sem["area"] *= 6.201172**2 # [nm²]
    sem_area = sem["area"]
    sem_ids = sem["id"]

    for file in os.listdir(loaddir):
        if re.fullmatch(r"([A-Z]{1}[0-9]{1})(_corr.csv)$", file) is not None:
            files.append(file)
            ids.append(file[0:2])

    for r in range(len(ids)):
        for s in range(len(sem_ids)):
            if ids[r] == sem_ids[s]:
                plot_ids.append(name+'_'+ids[r])
                plot_files.append(files[r])
                plot_areas.append(sem_area[s])

plot_ids = np.array(plot_ids)
plot_files = np.array(plot_files)
plot_areas = np.array(plot_areas)


sorted = np.argsort(plot_areas)
plot_areas = plot_areas[sorted]
plot_files = plot_files[sorted]
plot_ids = plot_ids[sorted]

yticks = []
labels_waterfall = []
max_int = 0

#y_pos = dist/dist.max()
y_pos = np.linspace(0,1,len(plot_areas))*8
print(y_pos)

maximum = 0
for f in plot_files:
    wl, counts = np.loadtxt(open(loaddir+f, "rb"), delimiter=",", skiprows=6, unpack=True)
    maximum = np.max([maximum,counts.max()])

cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 1, len(plot_files))]

fig, ax = newfig(0.9,3.0)
ax.axvline(532,color='black', linestyle='--', linewidth=0.5)

for i, f in zip(range(len(plot_files)),plot_files):

    wl, counts = wl, counts = np.loadtxt(open(loaddir+f, "rb"), delimiter=",", skiprows=6, unpack=True)
    mask = (wl >= minwl) & (wl <= maxwl)
    wl = wl[mask]
    counts = counts[mask]
    #counts -= counts[-1]
    counts/=maximum

    ax.plot(wl,counts+y_pos[i], linewidth=1.0,color = colors[i])
    #ax.arrow(900,y_pos[i],10,0,color=colors[i],linewidth=2.0)
    ax.annotate("", xy=(920, y_pos[i]), xytext=(902, y_pos[i]), arrowprops = dict(color=colors[i],width=.4,headwidth=2.0,headlength=2.0))

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

