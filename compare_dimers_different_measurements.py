
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
import string

# sns.set_style("ticks", {'axes.linewidth': 1.0,
#                         'xtick.direction': 'in',
#                         'ytick.direction': 'in',
#                         'xtick.major.size': 3,
#                         'xtick.minor.size': 1.5,
#                         'ytick.major.size': 3,
#                         'ytick.minor.size': 1.5
#                         })




path = '/home/sei/Auswertung/p45_vergleich/'

diameter = 90 #nm


try:
    nmpx = np.loadtxt(path+"nmppx")
except:
    raise RuntimeError("nmppx not found!")


print(nmpx)

xerr = 3*nmpx

diameter = 90 #nm

maxwl = 950
minwl = 450

with open(path+'rejected_approved.txt', "r") as f:
    lines = f.read().split(' ')

approved = []
for word in lines:
    if word[0:1] is '\n':
        break
    if word[0] in string.ascii_uppercase:
        approved.append(word)


dist = np.array([])
d = pd.read_csv(path + 'p45m_did5_particles_SEM.csv', delimiter=',')
for i in range(d.shape[0]):
    if d.id[i] in approved:
        dist = np.append(dist, d.dist[i])

ind = dist.argsort()
dist = dist[ind]
approved = np.array(approved)
approved = approved[ind]

print(approved)
print(dist)

peakfiles_normal = []
specs_normal = []
spec_path = '/home/sei/Spektren/p45m_did5_par5/specs/'
peak_path = '/home/sei/Spektren/p45m_did5_par5/fitted/'
for id in approved:
    peakfiles_normal.append(peak_path + id+'.csv')
    specs_normal.append(spec_path + id+'.csv')

peakfiles_oxidized = []
specs_oxidized = []
spec_path = '/home/sei/Spektren/p45m_did5_par7/specs/'
peak_path = '/home/sei/Spektren/p45m_did5_par7/fitted/'
for id in approved:
    peakfiles_oxidized.append(peak_path + id+'.csv')
    specs_oxidized.append(spec_path + id+'.csv')

peakfiles_reduced = []
specs_reduced = []
spec_path = '/home/sei/Spektren/p45m_did5_par12/specs/'
peak_path = '/home/sei/Spektren/p45m_did5_par12/fitted/'
for id in approved:
    peakfiles_reduced.append(peak_path + id+'.csv')
    specs_reduced.append(spec_path + id+'.csv')



maxwl_fit = 850
minwl_fit = 450

cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 1, len(specs_normal))]

size = figsize(1.0)
fig, axes = plt.subplots(figsize=(size[0]*1.0, size[0]*0.8),ncols=3, sharey=True)

def plot_waterfall(ax,specs,peakfiles,sidelabels=False):
    ax.axvline(500,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
    ax.axvline(600,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
    ax.axvline(700,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
    ax.axvline(800,color='black', linestyle='--', linewidth=0.5,alpha=0.5)


    peaks = np.zeros(len(specs))
    peaks_err = np.zeros(len(specs))
    widths = np.zeros(len(specs))
    widths_err = np.zeros(len(specs))
    dist_fac = 0.25
    yticks = []
    labels_waterfall = []
    max_int = 0

    #y_pos = dist/dist.max()
    y_pos = np.linspace(0,1,len(dist))
    print(y_pos)

    maximum = 0
    for spec in specs:
        wl, counts = np.loadtxt(open(spec, "rb"), delimiter=",", skiprows=16, unpack=True)
        mask_fit = (wl >= minwl_fit) & (wl <= maxwl_fit)
        wl = wl[mask_fit]
        filtered = savgol_filter(counts, 51, 0)
        maximum = np.max([maximum,filtered.max()])

    for i, spec in zip(range(len(specs)),specs):
        print(spec)

        wl, counts = np.loadtxt(open(spec, "rb"), delimiter=",", skiprows=16, unpack=True)
        wl = wl[mask_fit]
        filtered = savgol_filter(counts, 51, 0)

        filtered = filtered[mask_fit]

        filtered -= filtered.min()
        #filtered /= filtered.max()
        filtered /= maximum
        filtered /= 4

        #ax.plot(wl,filtered+i*dist_fac, linewidth=1.0,color = colors[i])
        ax.plot(wl,filtered+y_pos[i], linewidth=1.0,color = colors[i])

        ax.set_xlim([minwl_fit,maxwl_fit])

        #yticks.append(filtered[0] + i *dist_fac)
        #yticks.append(filtered[0] + y_pos[i])
        yticks.append(y_pos[i])

        #max_int = np.max([max_int,filtered.max()+i*dist_fac])
        max_int = np.max([max_int,filtered.max()+y_pos[i]])

        peakdata = np.loadtxt(open(peakfiles[i], "rb"), delimiter=",", skiprows=1, unpack=False)
        #print(peakdata[:,0])

        # x0s = data[:, 0]
        # x0s_err = data[:, 1]
        # amps = data[:, 2]
        # amps_err = data[:, 3]
        # sigmas = data[:, 4]
        # sigma_err = data[:, 5]
        # c = data[0, 6]
        # c_err = data[0, 7]

        if len(peakdata.shape) > 1:
            peaks[i] = peakdata[0,0]
            peaks_err[i] = peakdata[0,1]
            widths[i] = peakdata[0,4]
            widths_err[i] = peakdata[0,5]
        else:
            peaks[i] = peakdata[0]
            peaks_err[i] = peakdata[1]
            widths[i] = peakdata[4]
            widths_err[i] = peakdata[5]

        #ax.scatter(peak_wl,filtered[np.argmax(filtered)]+i*dist_fac,s=20,marker="x",color = colors[i])
        ax.scatter(peaks[i], filtered[np.abs(wl-peaks[i]).argmin()] + y_pos[i], s=20, marker="x", color=colors[i])

        print(" peak wl:"+str(peaks[i]))
        ind = 0#np.argmin(wl - wl.max()*0.8)
        #plt.text(wl[ind],filtered[ind]*1.1+i*0.3,str(round(dist[i],1))+'nm')
        labels_waterfall.append(str(round(dist[i],1))+'nm')

        #ax.set_ylabel(r'$I_{df}\, /\, a.u.$')
        ax.set_xlabel(r'$\lambda\, /\, nm$')
        ax.set_xticks([500,600,700,800])

        if sidelabels:
            ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off', labelright='on')
            ax.set_yticks(yticks)
            ax.set_yticklabels(labels_waterfall)
        else:
            ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off', labelright='off')


plot_waterfall(axes[0],specs_normal,peakfiles_normal)
axes[0].set_title('a)')
axes[0].set_ylabel(r'$I_{df}\, /\, a.u.$')
plot_waterfall(axes[1],specs_oxidized,peakfiles_oxidized)
axes[1].set_title('b)')
plot_waterfall(axes[2],specs_reduced,peakfiles_reduced,sidelabels=True)
axes[2].set_title('c)')

#ax.set_ylim([0, (len(pics)+1)*dist_fac*1.1])
#ax.set_ylim([0, max_int*1.05])


plt.tight_layout()
#plt.show()
plt.savefig(path + "compare_waterfall.pdf", dpi= 400)
plt.savefig(path + "compare_waterfall.pgf")
plt.savefig(path + "compare_waterfall.png", dpi= 400)
plt.close()




def get_peakdata(peakfiles):

    peaks = np.zeros(len(peakfiles))
    amps = np.zeros(len(peakfiles))

    for i, spec in zip(range(len(peakfiles)),peakfiles):

        peakdata = np.loadtxt(open(peakfiles[i], "rb"), delimiter=",", skiprows=1, unpack=False)
        if len(peakdata.shape) > 1:
            peaks[i] = peakdata[0, 0]
            amps[i] = peakdata[0,2]
        else:
            peaks[i] = peakdata[0]
            amps[i] = peakdata[2]
    return peaks, amps


fig, ax = newfig(0.9)
peaks, amps = get_peakdata(peakfiles_normal)
ax.plot(dist/diameter,peaks, color='C0',linewidth=0.5)
ax.scatter(dist/diameter,peaks, s=30, marker=".", color='C0',label='Ursprünglich')
peaks, amps = get_peakdata(peakfiles_oxidized)
ax.plot(dist/diameter,peaks, color='C1',linewidth=0.5)
ax.scatter(dist/diameter,peaks, s=30, marker=".", color='C1',label='Oxidiert')
peaks, amps = get_peakdata(peakfiles_reduced)
ax.plot(dist/diameter,peaks, color='C2',linewidth=0.5)
ax.scatter(dist/diameter,peaks, s=30, marker=".", color='C2',label='Reduziert')
ax.set_ylabel("Resonanz-Wellenlänge / nm")
ax.set_xlabel("Abstand / Durchmesser")
plt.legend()
plt.tight_layout()
plt.savefig(path + "compare_peaks.pdf", dpi= 400)
plt.savefig(path + "compare_peaks.pgf")
plt.savefig(path + "compare_peaks.png", dpi= 400)
plt.close()
