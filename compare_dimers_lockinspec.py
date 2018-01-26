
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

sns.set_style("ticks", {'axes.linewidth': 1.0,
                        'xtick.direction': 'in',
                        'ytick.direction': 'in',
                        'xtick.major.size': 3,
                        'xtick.minor.size': 1.5,
                        'ytick.major.size': 3,
                        'ytick.minor.size': 1.5
                        })
# sample = "p52m"
# arrays = ["dif5"]
# suffix = "_par"

sample = "p45m"
# #arrays = ["did5","did6"]
arrays = ["did5"]
suffix = "_par5"

# sample = "p41m"
# arrays = ["dif3","dif4","dif5"]
# suffix = "_par"

#sample = "p52m"
#arrays = ["dif0","dif1","dif3","dif5"]
#suffix = ""

path = '/home/sei/Auswertung/'+sample+'_lockin/'

#nmpx = 5.17  # nm/px
try:
    nmpx = np.loadtxt(path+"nmppx")
except:
    raise RuntimeError("nmppx not found!")

print(nmpx)


#savedir = array + '/plots/'
#fname = path + sample + "_" + array + ".jpg"

maxwl = 950
minwl = 450

def lorentz(x, amplitude, xo, sigma):
    g = amplitude * np.square(sigma / 2) / (np.square(sigma / 2) + np.square(x - xo))
    return g.ravel()

def lorentzSum(x,p):
    n = int(len(p) / 3)
    amp = p[:n]
    x0 = p[n:2 * n]
    sigma = p[2 * n:3 * n]
    res = lorentz(x, amp[0], x0[0], sigma[0])
    for i in range(len(amp) - 1):
        res += lorentz(x, amp[i + 1], x0[i + 1], sigma[i + 1])
    #res += p[-1]
    return res

def err_fun(x,y,p):
    fit = lorentzSum(x, p)
    diff = np.abs(y - fit)
    return np.sum(diff)


pics = np.array([])
dist = np.array([])
labels = np.array([])
area = np.array([])
peakfiles = []
specs = []

sem_ids = []
spec_ids = []
sem_inds = []
spec_inds = []


print('-> loading files')
for a in arrays:

    d = pd.read_csv(path+a+'/'+sample+'_'+a+'_particles_SEM.csv', delimiter=',')
    for i in range(d.shape[0]):
        if d.particles[i] >= 2.0:
            if d.rdiff[i] < 0.5:
                pics = np.append(pics,path+a+'/plots/'+d.id[i]+'.png')
                #specs = np.append(specs,'/home/sei/Spektren/' + sample + '_' + a + suffix +'/specs/'+ d.id[i] + '.csv')
                dist = np.append(dist,d.dist[i])
                labels = np.append(labels,a+'_'+d.id[i])
                #areadiff = np.append(areadiff,d.areadiff[i])
                area = np.append(area, d.area[i])
                sem_ids.append(d.id[i])

    peakpath = '/home/sei/Spektren/'+sample+'_lockin_'+a+'/'
    with os.scandir(peakpath) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir():
                if re.fullmatch(r"([a-zA-Z]{1}[0-9]{1})", entry.name) is not None:
                    peakfiles.append(peakpath+entry.name+'/'+entry.name+"_peaks.csv")
                    specs.append(peakpath+entry.name+'/'+entry.name+"_lockin.csv")
                    spec_ids.append(entry.name)

print(sem_ids)

for i,sem_id in enumerate(sem_ids):
    for j,spec_id in enumerate(spec_ids):
        if sem_id == spec_id:
            sem_inds.append(i)
            spec_inds.append(j)

sem_inds = np.array(sem_inds)
spec_inds = np.array(spec_inds)
sem_ids = np.array(sem_ids)
spec_ids = np.array(spec_ids)

print(sem_ids[sem_inds])
print(spec_ids[spec_inds])
ids = spec_ids[spec_inds]

specs = np.array(specs)
peakfiles = np.array(peakfiles)

specs = specs[spec_inds]
peakfiles = peakfiles[spec_inds]

pics = pics[sem_inds]
dist = dist[sem_inds]
labels = labels[sem_inds]
area = area[sem_inds]


ind = dist.argsort()
pics = pics[ind]
peakfiles = peakfiles[ind]
specs = specs[ind]
labels = labels[ind]
dist = dist[ind]
area = area[ind]
area = area*nmpx**2
ids = ids[ind]
print("-> Distances of loaded dimers:")
print(dist)
#print(pics)
#print(specs)

size = [3]
fig = plt.figure(figsize=(size[0],size[0]*0.5*len(pics)))
#fig = plt.figure()
gs1 = gridspec.GridSpec(len(pics),2,width_ratios=[3,1])

for i, pic, spec in zip(range(len(pics)),pics,specs):
    wl, amp, phase = np.loadtxt(open(spec, "rb"), delimiter=",", skiprows=1, unpack=True)
    mask = (wl >= minwl) & (wl <= maxwl)

    amp -= amp.min()
    amp = amp/amp.max()
    #filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    ax = plt.subplot(gs1[2*i])
    ax.plot(wl[mask],amp[mask])
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    #ax.set_ylim([0, 1.1])
    ax.set_xlim([minwl, maxwl])
    ax.set_ylabel(labels[i])
    #ax.set_axis_off()
    #ax.text(wl.min(),counts.max()-0.2, spec[], color="white", fontsize=8)

    ax = plt.subplot(gs1[2 * i + 1])
    img = np.asarray(PIL.Image.open(pic))
    #img = 255 - img
    ax.imshow(img,cmap="gray")#, aspect='auto', extent=(0.4, 0.6, .5, .7), zorder=-1)
    ax.set_axis_off()

gs1.update(wspace=0.1, hspace=0.1) # set the spacing between axes.
#plt.show()
plt.savefig(path + "dimer_overview.pdf", dpi= 400)
plt.close()


maxwl_fit = 800
minwl_fit = 500
mask_fit = (wl >= minwl_fit) & (wl <= maxwl_fit)

fig = plt.figure(figsize=(size[0],size[0]*2.5))
peaks = np.zeros(len(specs))
peaks_err = np.zeros(len(specs))

for i, peakfile in enumerate(peakfiles):
    p = np.loadtxt(open(peakfile, "rb"), delimiter=",")
    #peaks[i] = p[0]-p[1]
    if len(p) > 0:
        peaks[i] = p[0]
    else:
        peaks[i] = p
    print(peakfile+" peak wl:"+str(peaks[i]))



for i, peakfile, spec in zip(range(len(pics)),peakfiles,specs):
    wl, amp, phase = np.loadtxt(open(spec, "rb"), delimiter=",", skiprows=1, unpack=True)
    mask = (wl >= minwl) & (wl <= maxwl)
    filtered = signal.savgol_filter(amp,71,1)
    filtered /= filtered.max()
    plt.plot(wl[mask],filtered[mask]+i*0.3, linewidth=1.0)
    plt.xlim([minwl,maxwl])

    wl = wl[mask_fit]
    filtered = filtered[mask_fit]

    p = np.loadtxt(open(peakfile, "rb"), delimiter=",")
    plt.scatter(p[0],filtered[np.argmax(filtered)]+i*0.3,s=20,marker="x")


plt.ylim([0, (len(pics)+1)*0.3*1.1])
#plt.show()
plt.savefig(path + "dimer_waterfall.pdf", dpi= 400)
plt.close()


fig, ax1 = plt.subplots()
popt,cov = np.polyfit(dist,peaks,1,cov=True)

fit_fn = np.poly1d(popt)
ax1.plot(dist, fit_fn(dist), '-',color='C0')

perr = np.sqrt(np.diag(cov)) # standardabweichung

#print("structure #: {0:d}/{1:d} {4:s} ,dist #: {2:d}/{3:d}, total #: {6:d}/{5:d}".format(l + 1, len(structures), k + 1, len(dists), prefixes[l], n_total, at))

#plt.suptitle('$({0:3.3f}\pm{1:3.3f} )*x + ({2:3.3f}\pm{3:3.3f}$)'.format(popt[0],perr[0],popt[1],perr[1]))
#plt.suptitle('({0:3.3f}+-{1:3.3f} )*x + ({2:3.3f}+-{3:3.3f})'.format(popt[0],perr[0],popt[1],perr[1]), y=1.05, fontsize=18)

#plt.scatter(dist,peaks)
(_, caps, _) = ax1.errorbar(dist, peaks, xerr=3.0, fmt='.', elinewidth=0.5, markersize=6, capsize=4,color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)

ax1.set_ylabel("peak wavelength / nm",color='C0')
ax1.set_xlabel("gap width / nm")

texts = []
for x, y, s in zip(dist, peaks,ids):
    texts.append(ax1.text(x,y,s))

adjust_text(texts)

ax2 = ax1.twinx()
ax2.scatter(dist,area,s=20,marker="+",color='C1')
ax2.set_ylabel('area / nm²',color='C1')

#plt.tight_layout()
plt.savefig(path + "dimer_peaks.pdf", dpi= 300)
plt.close()


fig = plt.figure()
plt.scatter(area,peaks,s=20)
plt.xlabel('area / nm²')
plt.ylabel("peak wavelength / nm")
#plt.tight_layout()
plt.savefig(path + "area_peaks.pdf", dpi= 300)
plt.close()


#n, bins, patches = plt.hist(area, 30, alpha=0.75)
# n, bins, patches = plt.hist(area, len(area)/2,alpha=0.75)
# plt.xlabel('$Area\ /\ nm^{2}$')
# plt.ylabel('Occurance')
# plt.title(r'$\mathrm{Histogram\ of\ Area}$')
# #plt.tight_layout()
# plt.savefig(path + "area_hist.png", dpi= 300)
# plt.close()




# try:
#     os.mkdir(path+"overview/")
# except:
#     pass
#
# files = []
# for file in os.listdir(path+savedir):
#     if re.fullmatch(r"([A-Z]{1}[1-9]{1})(.png)$", file) is not None:
#         files.append(file)
#
# files.sort()
#
# #n = int(np.sqrt(len(files)))
# nx = 5
# ny = 5
#
# #fig, axs = plt.subplots(n,n, figsize=figsize(1.0))
# #fig.subplots_adjust(hspace = .001, wspace=.001)
#
# size = figsize(1.0)
# fig = plt.figure(figsize=(size[0],size[0]))
# gs1 = gridspec.GridSpec(nx, ny)
# #indices = np.arange(0,len(files),1)
# c=0
# for ix in range(nx):
#     for iy in range(ny):
#         i = (ix)+(iy*ny)
#         print(i)
#         file = files[i]
#         img = np.asarray(PIL.Image.open(path + savedir + file))
#         img = 255 - img
#         ax = plt.subplot(gs1[c])
#         ax.imshow(img)
#         ax.text(1, img.shape[1] - 1, file[:-4], color="white", fontsize=8)
#         ax.set_axis_off()
#         # ax.set_title(file[:-4])
#         c += 1
#
# # for i,file in enumerate(files):
# #     print(i)
# #     img = np.asarray(PIL.Image.open(path+savedir+file))
# #     img = 255-img
# #     ax = plt.subplot(gs1[i])
# #     ax.imshow(img)
# #     ax.text(1,img.shape[1]-1,file[:-4],color="white",fontsize=8)
# #     ax.set_axis_off()
# #     #ax.set_title(file[:-4])
#
# #plt.tight_layout(.5)
# gs1.update(wspace=0.1, hspace=0.1) # set the spacing between axes.
# #plt.show()
# plt.savefig(path+"overview/" + "sem_overview.png", dpi= 300)
# plt.savefig(path+"overview/" + "sem_overview.pgf")
# plt.close()