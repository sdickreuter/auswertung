__author__ = 'sei'

import os
import pickle
import numpy as np
import scipy

from plotsettings import *

#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
import pandas as pd

class Resonances():
    def __init__(self, id, amp, x0, sigma):
        self.id = id
        self.amp = amp
        self.x0 = x0
        self.sigma = sigma


substrate = '2C1'
path = '/home/sei/Auswertung/'+substrate+'/'
sample = '2C1_75hept_B2'
#sample = '2C1_100hex_C2'
#sample = '2C1_150hept_B2'
#sample = '2C1_150hex_B3'
#sample = '2C1_200hex_A2'
#sample = '2C1_100hex_C2'

sem = pd.read_csv(path+sample+"/particles_SEM.csv", sep=',')
sem["area"] *= 6.201172**2 # [nm²]

data532 = pd.read_csv(path+sample+"/peaks_532nm.csv", sep=',')

dataraman = pd.read_csv(path+sample+"/peaks_Raman.csv", sep=',')

with open(path+sample+r"/resonances.pickle", "rb") as input_file:
    resonances = pickle.load(input_file)

area = np.array(sem["area"])
int532 = np.array(data532["max"])
ramanmax = np.array(dataraman["max"])
raman = np.array(dataraman["max_gauss"])
ramanerr = np.array(dataraman["stddev"])


ids = np.array(np.repeat(None,len(resonances)),dtype=object)
for i in range(len(resonances)):
    ids[i] = resonances[i].id

area_ind = []
int532_ind = []
raman_ind = []
res_ind = []
for i in range(len(ids)):
    area_buf = np.where(sem["id"] == ids[i])[0]
    int532_buf = np.where(data532["id"] == ids[i])[0]
    raman_buf = np.where(dataraman["id"] == ids[i])[0]
    if len(raman_buf) > 0:
        if len(area_buf) > 0:
            if len(int532_buf) > 0:
                raman_ind.append(raman_buf[0])
                area_ind.append(area_buf[0])
                int532_ind.append(int532_buf[0])
                res_ind.append(i)

area_ind = np.array(area_ind)
int532_ind = np.array(int532_ind)
raman_ind = np.array(raman_ind)
res_ind = np.array(res_ind)

area = area[area_ind]
int532 = int532[int532_ind]
ramanmax = ramanmax[raman_ind]
raman = raman[raman_ind]
ramanerr = ramanerr[raman_ind]
resonances = resonances[res_ind]
ids = ids[res_ind]

ordered = np.argsort(area).ravel()

ids = ids[ordered]
area = area[ordered]/1000
int532 = int532[ordered]
ramanmax = ramanmax[ordered]
raman = raman[ordered]
ramanerr =ramanerr[ordered]
resonances = resonances[ordered]

try:
    os.mkdir(path+sample+"/graphs_2/")
except:
    pass

ppath = path+sample+"/graphs_2/"

name = ppath + "raman.png"
name_pgf = name[:-4] + ".pgf"
fig = newfig(0.9)
#plt.errorbar(area,raman,yerr=ramanerr,fmt='o',markersize=3)
plt.plot(area,raman,'o')
#plt.title('Raman Intensity')
plt.ylabel(r'$I_{\nu}\, /\, a.u.$')
plt.xlabel(r'$Area\, / \,10^3 nm^2$')
#sns.despine()
#plt.show()
plt.tight_layout()
# plt.savefig(name, dpi=300)
# plt.savefig(name_pgf)
plt.savefig(name[:-4] + ".eps",dpi=1200)
plt.close()

name = ppath + "int532.png"
name_pgf = name[:-4] + ".pgf"
fig = newfig(0.9)
plt.plot(area,int532,"o")
#plt.title('Darkfield Intensity at 532nm')
plt.ylabel(r'$I_{df}\, / \, a.u.$')
plt.xlabel(r'$Area\, /\, 10^3 nm^2$')
#sns.despine()
#plt.show()
plt.tight_layout()
# plt.savefig(name, dpi=300)
# plt.savefig(name_pgf)
plt.savefig(name[:-4] + ".eps",dpi=1200)
plt.close()

name = ppath + "raman532corr.png"
name_pgf = name[:-4] + ".pgf"
fig = newfig(0.9)
plt.plot(int532,raman,"o")
#plt.title('Correlation Raman/Darkfield')
plt.ylabel(r'$I_{\nu}\, / \,a.u.$')
plt.xlabel(r'$I_{df}\, @\,532nm\, / \,a.u.$')
#sns.despine()
#plt.show()
plt.tight_layout()
# plt.savefig(name, dpi=300)
# plt.savefig(name_pgf)
plt.savefig(name[:-4] + ".eps",dpi=1200)
plt.close()


amps = []
for i in range(len(resonances)):
    amp = resonances[i].amp
    for j in range(len(amp)):
        amps.append(amp[j])

amps = np.array(amps)
#cols = sns.color_palette(n_colors=len(resonances))
cols = sns.color_palette("husl", len(resonances))
fig, ax1 = plt.subplots()
ax1.set_xlabel(r'$Area\, / \,10^3 nm^2$')
ax1.set_ylabel(r'$\lambda\, / \,nm$')
ax1.set_ylim((450,950))
#ax1.set_xlim((np.min(area)-1,np.max(area)+10))
ax2 = ax1.twinx()
ax2.set_ylabel(r'$I_{\nu}\, / \, a.u.$')
ax2.plot(area,raman,'k.')
for i in range(len(resonances)):
    amp = resonances[i].amp
    x0 = resonances[i].x0
    sigma = resonances[i].sigma
    #quality = amp/sigma
    alpha = (amp-np.min(amps))/np.max(amps)
    for j in range(len(x0)):
        #if quality[j] > 0.0002:
        #plt.plot(area[i],x0[j],"o",color=cols[i],alpha=alpha[j])
        #plt.vlines(area[i], x0[j]+alpha[j]*10, x0[j]-alpha[j]*10,color=cols[i] )
        if alpha[j] > 0.02:
            ax1.hlines(x0[j], area[i]+alpha[j]*1, area[i]-alpha[j]*1,color=cols[i] )


#plt.show()
plt.savefig(ppath + "resonances_raman.png", format='png')
plt.close()


def gauss(x, amplitude, xo, fwhm):
    sigma = fwhm / 2.3548
    xo = float(xo)
    g = amplitude * np.exp(-np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))
    return g.ravel()

resfactor = np.zeros(len(resonances))
for i in range(len(resonances)):
    amp = resonances[i].amp
    x0 = resonances[i].x0
    sigma = resonances[i].sigma
    fac = (amp-np.min(amps))/np.max(amps)
    #resfactor[i] = np.sum(gauss(x0,1,532,100))
    resfactor[i] = np.sum(amp*sigma)


plt.plot(resfactor,raman,"o")
plt.title('Raman vs Number of Resonances')
plt.ylabel(r'$I_{\nu}\, / \,a.u.$')
plt.xlabel(r'$Number\,of\,Resonances$')
#sns.despine()
#plt.show()
plt.savefig(ppath + "resfactor.png", format='png')
plt.close()


specpath = '/home/sei/Spektren/'+substrate+'/'+sample+'/specs/'
picpath =  '/home/sei/Auswertung/'+substrate+'/'+sample+'/plots/'

n = len(ids)
fig, axes = plt.subplots(nrows=n,ncols=3,figsize=(15, 120))
rmax = np.max(raman)

for i in range(n):
    ax0, ax1, ax2 = axes[i,:]
    #spec = pd.read_csv(specpath+ids[i]+"_corr.csv", sep=',',)
    wl, counts = np.loadtxt(open(specpath+ids[i]+"_corr.csv", "rb"), delimiter=",", skiprows=7, unpack=True)
    pic = scipy.misc.imread(picpath+ids[i]+'.png')
    ax0.imshow(1-pic, interpolation='nearest')
    ax0.axis('off')
    #ax0.set_title('Overlapping objects')
    ax1.plot(wl, counts)
    #ax1.set_title('Distances')
    ax2.hlines(0,raman[i]/rmax,0,linewidths=20)
    ax2.set_xlim(0,1)
    ax2.axis('off')


#plt.show()

plt.savefig(ppath + "overview.png", format='png')
plt.close()


n = len(ids)
specmap = np.zeros((len(counts),n))
x = np.zeros(n)

for i in range(n):
    wl, counts = np.loadtxt(open(specpath+ids[i]+"_corr.csv", "rb"), delimiter=",", skiprows=7, unpack=True)
    specmap[:,i] = counts
    x[i] = i

sns.set_style("whitegrid")
sns.heatmap(specmap,xticklabels=False,yticklabels=False,vmin=0,cmap="RdBu_r")
plt.plot(x+0.5,raman*20,'k.')
#plt.show()
plt.savefig(ppath + "specmap.png", format='png')
plt.close()


#
#
# n = len(ids)
# fig, axes = plt.subplots(nrows=n,ncols=2,figsize=(10, 120))
#
# for i in range(n):
#     ax0, ax1 = axes[i,:]
#     #spec = pd.read_csv(specpath+ids[i]+"_corr.csv", sep=',',)
#     wl, counts = np.loadtxt(open(specpath+ids[i]+"_corr.csv", "rb"), delimiter=",", skiprows=7, unpack=True)
#     pic = scipy.misc.imread(picpath+ids[i]+'.png')
#     ax0.imshow(1-pic, interpolation='nearest')
#     ax0.axis('off')
#     #ax0.set_title('Overlapping objects')
#     ax1.plot(wl, counts)
#     #ax1.set_title('Distances')
#     #for ax in axes:
#     #    ax.axis('off')
#     #fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
#     #                    right=1)
#
# #plt.show()
#
# plt.savefig(ppath + "overview.png", format='png')
# plt.close()
