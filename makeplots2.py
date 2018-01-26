__author__ = 'sei'

import os
import pickle
import numpy as np
import scipy

import matplotlib as mpl

mpl.use("pgf")

def figsize(scale):
    fig_width_pt = 336.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }

mpl.rcParams.update(pgf_with_latex)




import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("ticks", {'axes.linewidth': 0.5})
sns.set_context("paper", rc= {  "xtick.major.width":0.5,
                            "ytick.major.width":0.5,
                            "xtick.minor.width":0.5,
                            "ytick.minor.width":0.5})
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

class Resonances():
    def __init__(self, id, amp, x0, sigma):
        self.id = id
        self.amp = amp
        self.x0 = x0
        self.sigma = sigma


path = '/home/sei/Auswertung/8/'
sample = 'D3'

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

try:
    os.mkdir(path+sample+"/graphs/")
except:
    pass

ppath = path+sample+"/graphs/"

specpath = '/home/sei/Spektren/8/8_'+sample+'_horzpol/specs/'

n = len(ids)
wl, counts = np.loadtxt(open(specpath+ids[0]+"_corr.csv", "rb"), delimiter=",", skiprows=7, unpack=True)
specmap = np.zeros((len(counts),n))
x = np.zeros(n)

for i in range(n):
    wl, counts = np.loadtxt(open(specpath+ids[i]+"_corr.csv", "rb"), delimiter=",", skiprows=7, unpack=True)
    specmap[:,i] = counts/np.max(counts)
    x[i] = i

sns.set_style("whitegrid")
sns.heatmap(specmap,xticklabels=False,yticklabels=False,vmin=0,cmap="RdBu_r")
#plt.show()
plt.savefig(ppath + "specmap.png", format='png')
plt.close()

plt.plot(dist,peak,"o")
plt.title('Peak Wavelength')
plt.ylabel(r'$\lambda_{max}\/[nm]$')
plt.xlabel(r'$Distance \/ [nm]$')
#sns.despine()
#plt.show()
plt.savefig(ppath + "dist.png", format='png')
plt.close()

x0 = np.zeros(0)
for res in resonances:
    x0 = np.hstack( (x0,res.x0[0]) )

plt.plot(dist,x0,"o")
plt.title('max. Darkfield Intensity')
plt.ylabel(r'$I_{df}\/[a.u.]$')
plt.xlabel(r'$Distance \/ [nm]$')
#sns.despine()
#plt.show()
plt.savefig(ppath + "dist_res.png", format='png')
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
ax1.set_xlabel(r'$Area \/ 10^3 [nm^2]$')
ax1.set_ylabel(r'$\lambda\/[nm]$')
ax1.set_ylim((450,950))
#ax1.set_xlim((np.min(area)-1,np.max(area)+10))
ax2 = ax1.twinx()
ax2.set_ylabel(r'$I_{\nu}\/[a.u.]$')
ax2.plot(dist,peak,'k.')
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
            ax1.hlines(x0[j], dist[i]+alpha[j]*1, dist[i]-alpha[j]*1,color=cols[i] )


#plt.show()
plt.savefig(ppath + "resonances_raman.png", format='png')
plt.close()



# plt.plot(raman,int532,"o")
# plt.title('Correlation Raman/Darkfield')
# plt.ylabel(r'$I_{df}\/[a.u.]$')
# plt.xlabel(r'$I_{df}\/[a.u.]$')
# #sns.despine()
# #plt.show()
# plt.savefig(ppath + "raman532corr.png", format='png')
# plt.close()
#
#

#
#
# def gauss(x, amplitude, xo, fwhm):
#     sigma = fwhm / 2.3548
#     xo = float(xo)
#     g = amplitude * np.exp(-np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))
#     return g.ravel()
#
# resfactor = np.zeros(len(resonances))
# for i in range(len(resonances)):
#     amp = resonances[i].amp
#     x0 = resonances[i].x0
#     sigma = resonances[i].sigma
#     fac = (amp-np.min(amps))/np.max(amps)
#     #resfactor[i] = np.sum(gauss(x0,1,532,100))
#     resfactor[i] = np.sum(amp*sigma)
#
#
# plt.plot(resfactor,raman,"o")
# plt.title('Raman vs Number of Resonances')
# plt.ylabel(r'$I_{\nu}\/[a.u.]$')
# plt.xlabel(r'$Number of Resonances$')
# #sns.despine()
# #plt.show()
# plt.savefig(ppath + "resfactor.png", format='png')
# plt.close()
#
#
# specpath = '/home/sei/Spektren/2C1/'+sample+'/specs/'
# picpath =  '/home/sei/Auswertung/2C1/'+sample+'/plots/'
#
# n = len(ids)
# fig, axes = plt.subplots(nrows=n,ncols=3,figsize=(15, 120))
# rmax = np.max(raman)
#
# for i in range(n):
#     ax0, ax1, ax2 = axes[i,:]
#     #spec = pd.read_csv(specpath+ids[i]+"_corr.csv", sep=',',)
#     wl, counts = np.loadtxt(open(specpath+ids[i]+"_corr.csv", "rb"), delimiter=",", skiprows=7, unpack=True)
#     pic = scipy.misc.imread(picpath+ids[i]+'.png')
#     ax0.imshow(1-pic, interpolation='nearest')
#     ax0.axis('off')
#     #ax0.set_title('Overlapping objects')
#     ax1.plot(wl, counts)
#     #ax1.set_title('Distances')
#     ax2.hlines(0,raman[i]/rmax,0,linewidths=20)
#     ax2.set_xlim(0,1)
#     ax2.axis('off')
#
#
# #plt.show()
#

#
#
# #
# #
# # n = len(ids)
# # fig, axes = plt.subplots(nrows=n,ncols=2,figsize=(10, 120))
# #
# # for i in range(n):
# #     ax0, ax1 = axes[i,:]
# #     #spec = pd.read_csv(specpath+ids[i]+"_corr.csv", sep=',',)
# #     wl, counts = np.loadtxt(open(specpath+ids[i]+"_corr.csv", "rb"), delimiter=",", skiprows=7, unpack=True)
# #     pic = scipy.misc.imread(picpath+ids[i]+'.png')
# #     ax0.imshow(1-pic, interpolation='nearest')
# #     ax0.axis('off')
# #     #ax0.set_title('Overlapping objects')
# #     ax1.plot(wl, counts)
# #     #ax1.set_title('Distances')
# #     #for ax in axes:
# #     #    ax.axis('off')
# #     #fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
# #     #                    right=1)
# #
# # #plt.show()
# #
# # plt.savefig(ppath + "overview.png", format='png')
# # plt.close()
