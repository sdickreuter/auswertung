__author__ = 'sei'

import os
import pickle
import numpy as np
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pandas as pd
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.4)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.signal import savgol_filter

class Resonances():
    def __init__(self, id, amp, x0, sigma):
        self.id = id
        self.amp = amp
        self.x0 = x0
        self.sigma = sigma


path = '/home/sei/Auswertung/8/'
samples = ['D2','D3','C3','C4']

dists = np.zeros(0)
peaks = np.zeros(0)
specs = np.zeros(0)
names = np.zeros(0)

for sample in samples:
    sem = pd.read_csv(path+sample+"/particles_SEM.csv", sep=',')
    #sem["area"] *= 6.201172**2 # [nm²]

    df = pd.read_csv(path+sample+"/peak_wl.csv", sep=',')


    with open(path+sample+r"/resonances.pickle", "rb") as input_file:
        resonances = pickle.load(input_file)

    dist = np.array(sem["dist"])
    peak = np.array(df["peak_wl"])

    ids = np.array(np.repeat(None,len(resonances)),dtype=object)
    for i in range(len(resonances)):
        ids[i] = resonances[i].id

    dist_ind = []
    peak_ind = []
    res_ind = []
    for i in range(len(ids)):
        dist_buf = np.where(sem["id"] == ids[i])[0]
        peak_buf = np.where(df["id"] == ids[i])[0]
        if len(peak_buf) > 0:
            if len(dist_buf) > 0:
                if dist[dist_buf[0]] > 0.1:
                    dist_ind.append(dist_buf[0])
                    peak_ind.append(peak_buf[0])
                    res_ind.append(i)

    dist_ind = np.array(dist_ind)
    peak_ind = np.array(peak_ind)
    res_ind = np.array(res_ind)

    dist = dist[dist_ind]
    peak = peak[peak_ind]
    resonances = resonances[res_ind]
    ids = ids[res_ind]

    ordered = np.argsort(dist).ravel()

    ids = ids[ordered]
    dist = dist[ordered]
    peak = peak[ordered]
    resonances = resonances[ordered]

    names = np.hstack( (names,ids) )
    dists = np.hstack( (dists,dist) )
    peaks = np.hstack( (peaks,peak) )

    specpath = '/home/sei/Spektren/8/8_'+sample+'_horzpol/specs/'

    n = len(ids)

    for i in range(n):
        specs = np.hstack( (specs,specpath+ids[i]+"_corr.csv") )


try:
    os.mkdir(path+"/gathered/")
except:
    pass

ppath = path+"/gathered/"

ordered = np.argsort(dists).ravel()
names = names[ordered]
dists = dists[ordered]
peaks = peaks[ordered]
specs = specs[ordered]

f = open(ppath+"data.csv", 'w')
f.write("name,dist,peak,spec"+"\r\n")
for i in range(len(names)):
    f.write(str(names[i])+","+str(dists[i])+","+str(peaks[i])+","+str(specs[i])+"\r\n")

f.close()

n = len(names)
wl, counts = np.loadtxt(open(specs[0], "rb"), delimiter=",", skiprows=7, unpack=True)
specmap = np.zeros((len(counts),n))
x = np.zeros(n)

for i in range(n):
    wl, counts = np.loadtxt(open(specs[i], "rb"), delimiter=",", skiprows=7, unpack=True)
    specmap[:,i] = counts/np.max(counts)
    x[i] = i

sns.set_style("whitegrid")
sns.heatmap(specmap,xticklabels=False,yticklabels=False,vmin=0,cmap="RdBu_r")
#plt.show()
plt.savefig(ppath + "specmap.png", format='png')
plt.close()


fig = plt.figure()
ax = fig.gca(projection='3d')
cols = sns.color_palette("husl",len(dists))

verts = []
for i in range(len(dists)):
    wl, counts = np.loadtxt(open(specs[i], "rb"), delimiter=",", skiprows=7, unpack=True)
    counts = savgol_filter(counts, 21, 1)
    counts[0], counts[-1] = 0, 0
    #counts = counts-np.mean(counts[0:20])
    counts = counts/np.max(counts)
    verts.append(list(zip(wl, counts)))

poly = PolyCollection(verts, facecolors=cols)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=dists, zdir='y')

ax.set_xlabel('wavelength')
ax.set_xlim3d(400, 900)
ax.set_ylabel('distance')
ax.set_ylim3d(6, 100)
ax.set_zlabel('intensity')
ax.set_zlim3d(0, 5)

plt.show()


ind = np.array(np.where(dists<20))[0,:]
cols = sns.color_palette("husl",len(ind))

for i in range(len(ind)):
    wl, counts = np.loadtxt(open(specs[ind[i]], "rb"), delimiter=",", skiprows=7, unpack=True)
    counts = counts-np.mean(counts[0:20])
    counts = counts/np.max(counts)
    plt.plot(wl,counts*3+dists[ind[i]],color=cols[i])

#plt.show()
plt.savefig(ppath + "specs.png", format='png')
plt.close()


fig = plt.figure()
ax = fig.gca(projection='3d')

verts = []
for i in ind:
    wl, counts = np.loadtxt(open(specs[i], "rb"), delimiter=",", skiprows=7, unpack=True)
    counts = savgol_filter(counts, 21, 1)
    counts[0], counts[-1] = 0, 0
    #counts = counts-np.mean(counts[0:20])
    counts = counts/np.max(counts)
    verts.append(list(zip(wl, counts+i/10)))

poly = PolyCollection(verts, facecolors=cols)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=dists[ind], zdir='y')

ax.set_xlabel('wavelength')
ax.set_xlim3d(400, 900)
ax.set_ylabel('distance')
ax.set_ylim3d(6, 20)
ax.set_zlabel('intensity')
ax.set_zlim3d(0, 5)

plt.show()