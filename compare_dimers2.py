
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


# sns.set_style("ticks", {'axes.linewidth': 1.0,
#                         'xtick.direction': 'in',
#                         'ytick.direction': 'in',
#                         'xtick.major.size': 3,
#                         'xtick.minor.size': 1.5,
#                         'ytick.major.size': 3,
#                         'ytick.minor.size': 1.5
#                         })
# sample = "p52m"
# arrays = ["dif5"]
# suffix = "_par"

sample = "p45m"
arrays = ["did5"]
suffix = "_par5"
#suffix = "_heat"

# sample = "p41m"
# arrays = ["dif5"]
# suffix = "_par"


diameter = 90 #nm


path = '/home/sei/Auswertung/'+sample+'_dimerpaper/'


try:
    nmpx = np.loadtxt(path+"nmppx")
except:
    raise RuntimeError("nmppx not found!")


print(nmpx)

xerr = 3*nmpx


maxwl = 950
minwl = 450


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

approved = []
rejected = []

print('-> loading files')
for a in arrays:

    d = pd.read_csv(path+a+'/'+sample+'_'+a+'_particles_SEM.csv', delimiter=',')
    for i in range(d.shape[0]):
        if d.particles[i] >= 2.0:
            if (d.rdiff[i] < 0.27) & (d.area[i] > 9000):
            #if True:
                pics = np.append(pics,path+a+'/plots/'+d.id[i]+'.png')
                #specs = np.append(specs,'/home/sei/Spektren/' + sample + '_' + a + suffix +'/specs/'+ d.id[i] + '.csv')
                dist = np.append(dist,d.dist[i])
                labels = np.append(labels,a+'_'+d.id[i])
                #areadiff = np.append(areadiff,d.areadiff[i])
                area = np.append(area, d.area[i])
                sem_ids.append(d.id[i])
                approved.append(d.id[i])
            else:
                rejected.append(d.id[i])
        else:
            rejected.append(d.id[i])

    f = open(path + "rejected_approved.txt", 'w')
    f.write("approved: ")
    for id in approved:
        f.write(id+' ')
    f.write("\r\n")
    f.write("rejected: ")
    for id in rejected:
        f.write(id + ' ')
    f.close()

    spec_path = '/home/sei/Spektren/'+sample+'_'+a+suffix+'/specs/'
    peak_path = '/home/sei/Spektren/'+sample+'_'+a+suffix+'/fitted/'
    with os.scandir(spec_path) as it:
        for entry in it:
            if not entry.name.startswith('.') and not entry.is_dir():
                if re.fullmatch(r"([a-zA-Z]{1}[0-9]{1}(.csv))", entry.name) is not None:
                    peakfiles.append(peak_path+entry.name)
                    specs.append(spec_path+entry.name)
                    spec_ids.append(entry.name[:-4])

# print(sem_ids)
# print(spec_ids)

for i,sem_id in enumerate(sem_ids):
    for j,spec_id in enumerate(spec_ids):
        if sem_id == spec_id:
            sem_inds.append(i)
            spec_inds.append(j)


sem_inds = np.array(sem_inds)
spec_inds = np.array(spec_inds)
sem_ids = np.array(sem_ids)
spec_ids = np.array(spec_ids)
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
specs = specs[ind]
peakfiles = peakfiles[ind]
labels = labels[ind]
dist = dist[ind]
area = area[ind]
area = area*nmpx**2
print("-> Distances of loaded dimers:")
print(dist)
#print(pics)
#print(specs)

size = [3]
fig = plt.figure(figsize=(size[0],size[0]*0.5*len(pics)))
#fig = plt.figure()
gs1 = gridspec.GridSpec(len(pics),2,width_ratios=[3,1])

for i, pic, spec in zip(range(len(pics)),pics,specs):
    print(spec)
    wl, counts = np.loadtxt(open(spec, "rb"), delimiter=",", skiprows=16, unpack=True)
    mask = (wl >= minwl) & (wl <= maxwl)

    counts -= counts.min()
    counts = counts/counts.max()
    #filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    ax = plt.subplot(gs1[2*i])
    ax.plot(wl[mask],counts[mask])
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
plt.savefig(path + "dimer_overview.png", dpi= 400)
plt.close()


maxwl_fit = 850
minwl_fit = 450
mask_fit = (wl >= minwl_fit) & (wl <= maxwl_fit)

cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 1, len(specs))]

#fig = plt.figure(figsize=(size[0],size[0]*2.5))
fig, ax = newfig(0.5,2.0)

ax.axvline(500,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(600,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(700,color='black', linestyle='--', linewidth=0.5,alpha=0.5)

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
    wl = wl[mask_fit]
    filtered = savgol_filter(counts, 51, 0)
    maximum = np.max([maximum,filtered.max()])

for i, spec in zip(range(len(specs)),specs):

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

    peaks[i] = peakdata[0,0]
    peaks_err[i] = peakdata[0,1]

    widths[i] = peakdata[0,4]
    widths_err[i] = peakdata[0,5]

    #ax.scatter(peak_wl,filtered[np.argmax(filtered)]+i*dist_fac,s=20,marker="x",color = colors[i])
    ax.scatter(peaks[i], filtered[np.abs(wl-peaks[i]).argmin()] + y_pos[i], s=20, marker="x", color=colors[i])

    print(spec+" peak wl:"+str(peaks[i]))
    ind = 0#np.argmin(wl - wl.max()*0.8)
    #plt.text(wl[ind],filtered[ind]*1.1+i*0.3,str(round(dist[i],1))+'nm')
    labels_waterfall.append(str(round(dist[i],1))+'nm')


#print(peaks)
#print(peaks_err)
ax.set_ylabel(r'$I_{df}\, /\, a.u.$')
ax.set_xlabel(r'$\lambda\, /\, nm$')
#ax.set_ylim([0, (len(pics)+1)*dist_fac*1.1])
ax.set_ylim([0, max_int*1.05])

ax.tick_params(axis='y', which='both',left='off',right='off', labelleft='off', labelright='on')
ax.set_yticks(yticks)
ax.set_yticklabels(labels_waterfall)

plt.tight_layout()
#plt.show()
plt.savefig(path + "dimer_waterfall.pdf", dpi= 400)
plt.savefig(path + "dimer_waterfall.pgf")
plt.savefig(path + "dimer_waterfall.png", dpi= 400)
plt.close()


fig, ax1 = plt.subplots()


(_, caps, _) = ax1.errorbar(dist, peaks, xerr=xerr,yerr=peaks_err, fmt='.', elinewidth=0.5, markersize=6, capsize=4,color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)
#
# texts = []
# for x, y, s in zip(dist, peaks,labels):
#     texts.append(ax1.text(x,y,s[-2:]))
#
# adjust_text(texts, autoalign='y', only_move={'points':'y', 'text':'y'},
#             #arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
#             expand_points=(1.5, 1.75),
#             force_points=0.1)

#ax1.set_ylabel("Peak Wavelength / nm",color='C0')
ax1.set_ylabel("Peak Wavelength / nm")
ax1.set_xlabel("Gap Width / nm")

# ax2 = ax1.twinx()
# ax2.scatter(dist,area,s=20,marker="+",color='C1')
# ax2.set_ylabel('Area / nm²',color='C1')

plt.tight_layout()
plt.savefig(path + "dimer_peaks.pdf", dpi= 300)
plt.savefig(path + "dimer_peaks.pgf")
plt.savefig(path + "dimer_peaks.png", dpi= 400)
plt.close()


x = dist/diameter
#y = (peaks-575)/575
y = peaks
fit_fun = lambda x, a, tau,c: a * np.exp(-x/tau)+c
p0 = [y.max(),0.2,0]
popt, pcov = curve_fit(fit_fun, x, y, p0,sigma=peaks_err)
perr = np.sqrt(np.diag(pcov))

f = open(path + "exponential_fit.txt", 'w')
f.write("params: ")
for a in popt:
    f.write(str(a) + ' ')
f.write("\r\n")
f.write("errs: ")
for a in perr:
    f.write(str(a) + ' ')
f.close()

fig, ax1 = plt.subplots()

ax1.plot(x,fit_fun(x,popt[0],popt[1],popt[2]),color="C1",linewidth=0.75,linestyle='--',zorder=0)

(_, caps, _) = ax1.errorbar(x, y, xerr=xerr/diameter,yerr=peaks_err, fmt='.', elinewidth=0.5, markersize=6, capsize=4,color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)

#ax1.text(s=r"$y="+str(round(popt[0],2))+"\cdot e^{-x/"+str(round(popt[1],2))+"}+"+str(round(popt[2],2))+"$",xy=(0.15,fit_fun(0.15,popt[0],popt[1],popt[2])))#,xytext=(0.3,660))
ax1.text(0.23,640,s='$y='+str(int(round(popt[0])))+' \cdot e^{-x/'+str(round(popt[1],2))+'}+'+str(int(round(popt[2])))+'$')
#ax1.text(0.3,600,s='$y=1 e^{-x/1}+1$')


ax1.set_ylabel("Peak Wavelength / nm")

ax1.set_xlabel("Gap / Diameter")
plt.tight_layout()
plt.savefig(path + "dimer_peaks2.pdf", dpi= 300)
plt.savefig(path + "dimer_peaks2.pgf")
plt.savefig(path + "dimer_peaks2.png", dpi= 400)
plt.close()





fig = plt.figure()
plt.scatter(area,peaks,s=20)
plt.xlabel('area / nm²')
plt.ylabel("peak wavelength / nm")
plt.tight_layout()
plt.savefig(path + "area_peaks.pdf", dpi= 300)
plt.savefig(path + "area_peaks.png", dpi= 300)
plt.close()


fig, ax = plt.subplots()
for i in range(len(peakfiles)):
    peakdata = np.loadtxt(open(peakfiles[i], "rb"), delimiter=",", skiprows=1, unpack=False)
    print(peakdata[:,0])
    ax.scatter(dist[i], peakdata[0,0], s=20, marker="x", color='C0')
    ax.scatter(dist[i], peakdata[1,0], s=20, marker="o", color='C1')
    ax.scatter(dist[i], peakdata[2,0], s=20, marker="D", color='C2')


ax.set_ylabel("Peak Wavelength / nm")
ax.set_xlabel("Gap Width / nm")
plt.tight_layout()
plt.savefig(path + "all_peaks.pdf", dpi=300)
plt.savefig(path + "all_peaks.pgf")
plt.savefig(path + "all_peaks.png", dpi=400)
plt.close()


fig, ax = plt.subplots()
for i in range(len(peakfiles)):
    peakdata = np.loadtxt(open(peakfiles[i], "rb"), delimiter=",", skiprows=1, unpack=False)
    print(peakdata[:,0])
    sca1 = ax.scatter(dist[i], peakdata[0,2], s=20, marker="x", color='C0', label='Strange Mode')
    sca2 = ax.scatter(dist[i], peakdata[1,2], s=20, marker="o", color='C1', label='Quad. Mode')
    sca3 = ax.scatter(dist[i], peakdata[2,2], s=20, marker="D", color='C2', label='Dipole Mode')

ax.legend(handles=[sca1,sca2,sca3],edgecolor='black',frameon=True)
ax.set_ylabel("Amplitude / a.u.")
ax.set_xlabel("Gap Width / nm")
plt.tight_layout()
plt.savefig(path + "all_amps.pdf", dpi=300)
plt.savefig(path + "all_amps.pgf")
plt.savefig(path + "all_amps.png", dpi=400)
plt.close()

fig, ax = plt.subplots()
for i in range(len(peakfiles)):
    peakdata = np.loadtxt(open(peakfiles[i], "rb"), delimiter=",", skiprows=1, unpack=False)
    print(peakdata[:,0])
    sca3 = ax.scatter(dist[i], peakdata[2,2], s=20, marker="D", color='C2', label='Dipole Mode')
ax.set_ylabel("Amplitude / a.u.")
ax.set_xlabel("Gap Width / nm")
plt.tight_layout()
plt.savefig(path + "dipole_amps.pdf", dpi=300)
plt.savefig(path + "dipole_amps.pgf")
plt.savefig(path + "dipole_amps.png", dpi=400)
plt.close()

fig, ax = plt.subplots()
for i in range(len(peakfiles)):
    peakdata = np.loadtxt(open(peakfiles[i], "rb"), delimiter=",", skiprows=1, unpack=False)
    print(peakdata[:,0])
    sca2 = ax.scatter(dist[i], peakdata[1,2], s=20, marker="o", color='C1', label='Quad. Mode')
ax.set_ylabel("Amplitude / a.u.")
ax.set_xlabel("Gap Width / nm")
plt.tight_layout()
plt.savefig(path + "quad_amps.pdf", dpi=300)
plt.savefig(path + "quad_amps.pgf")
plt.savefig(path + "quad_amps.png", dpi=400)
plt.close()

#n, bins, patches = plt.hist(area, 30, alpha=0.75)
# n, bins, patches = plt.hist(area, len(area)/2,alpha=0.75)
# plt.xlabel('$Area\ /\ nm^{2}$')
# plt.ylabel('Occurance')
# plt.title(r'$\mathrm{Histogram\ of\ Area}$')
# #plt.tight_layout()
# plt.savefig(path + "area_hist.png", dpi= 300)
# plt.close()



fig, ax1 = plt.subplots()
(_, caps, _) = ax1.errorbar(dist, widths, xerr=xerr,yerr=widths_err, fmt='.', elinewidth=0.5, markersize=6, capsize=4,color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)
ax1.set_ylabel("Peak Width / nm")
ax1.set_xlabel("Gap Width / nm")
plt.tight_layout()
plt.savefig(path + "dimer_widths.pdf", dpi= 300)
plt.savefig(path + "dimer_widths.pgf")
plt.savefig(path + "dimer_widths.png", dpi= 400)
plt.close()


y = (peaks*1e9)**2 / (2*np.pi*2.998e8) *widths
y_err = (peaks*1e9)**2 / (2*np.pi*2.998e8) *widths_err + (peaks_err*1e9)**2 / (2*np.pi*2.998e8) *widths
fig, ax1 = plt.subplots()
(_, caps, _) = ax1.errorbar(dist, y, xerr=xerr,yerr=y_err, fmt='.', elinewidth=0.5, markersize=6, capsize=4,color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)
ax.set_yscale("log", nonposy='clip')
#plt.semilogy(dist,y,'.')
ax1.set_ylabel("Damping / 1/s")
ax1.set_xlabel("Gap Width / nm")
plt.tight_layout()
plt.savefig(path + "dimer_damping.pdf", dpi= 300)
plt.savefig(path + "dimer_damping.pgf")
plt.savefig(path + "dimer_damping.png", dpi= 400)
plt.close()


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