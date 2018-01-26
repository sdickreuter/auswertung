import numpy as np
from scipy import fft, ifft
from scipy.optimize import curve_fit

import pandas as pd
from plotsettings import *
import os
from scipy import ndimage
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib import cm

def gauss(x, amplitude, xo, fwhm):
    sigma = fwhm / 2.3548
    #g = amplitude* 1/np.sqrt(2*np.pi*np.power(sigma, 2)) * np.exp(-np.power(x - xo, 2) / (2 * np.power(sigma, 2)))
    g = amplitude * np.exp(-np.power(x - xo, 2) / (2 * np.power(sigma, 2)))
    return g.ravel()

def three_gauss(x,c, a0,a1,a2, xo0, xo1, xo2, fwhm0, fwhm1, fwhm2):
    g = gauss(x,a0,xo0,fwhm0)+gauss(x,a1,xo1,fwhm1)+gauss(x,a2,xo2,fwhm2)+c
    return g.ravel()

def lorentz(x, amplitude, xo, sigma):
    g = amplitude * np.power(sigma / 2, 2) / (np.power(sigma / 2, 2) + np.power(x - xo, 2))
    return g.ravel()

def three_lorentz(x,c, a0,a1,a2, xo0, xo1, xo2, fwhm0, fwhm1, fwhm2):
    g = lorentz(x,a0,xo0,fwhm0)+lorentz(x,a1,xo1,fwhm1)+lorentz(x,a2,xo2,fwhm2)+c
    return g.ravel()

def two_lorentz(x,c, a0,a1, xo0, xo1, fwhm0, fwhm1):
    g = lorentz(x,a0,xo0,fwhm0)+lorentz(x,a1,xo1,fwhm1)+c
    return g.ravel()


path = '/home/sei/Emre/'
file = 'RawData_TPL.csv'

savedir = path + 'plots/'
try:
    os.mkdir(savedir)
except:
    pass


df = pd.read_csv(path+file,delimiter=';',decimal=',')

wl = df['Wavelength']
sizes_ind = df.axes[1][1:-1]
sizes = np.array(df.axes[1][1:-1],dtype=np.int)

mask = (wl > 620) | (wl < 615)
mask1 = (wl > 620)
mask2 = (wl < 615)


# wl_ind1 = np.argmin( np.abs(wl-615))
# wl_ind2 = np.argmin( np.abs(wl-620))
#
# drop_ind = np.arange(wl_ind1,wl_ind2+1)
#
# df.drop(drop_ind,inplace=True)
# wl = df['Wavelength']


index = sizes_ind[0]


x = wl
y = df[index]
#p0 = [0, 0.15,0.1,0.1, 550, 613, 630, 150, 15, 15]
p0 = [0, 0.15,0.1, 550, 630, 150, 30]

#
# plt.plot(wl,df[index])
# plt.plot(wl,two_lorentz(x,p0[0],p0[1],p0[2],p0[3],p0[4],p0[5],p0[6]))
# plt.show()

img_fit = np.zeros((len(sizes_ind), len(wl)))
wl1 = np.zeros(len(sizes_ind))
wl2 = np.zeros(len(sizes_ind))

wl1_ind = np.zeros(len(sizes_ind),dtype = np.int)
wl2_ind = np.zeros(len(sizes_ind), dtype = np.int)

for i,index in enumerate(sizes_ind):
    x = wl[mask]
    y = df[index][mask]

    popt, pcov = curve_fit(two_lorentz, x, y, p0)
    p0 = popt

    newfig(0.9)
    plt.plot(x,y, linewidth=0.6,alpha=0.7,marker='.',linestyle='')
    plt.plot(wl,two_lorentz(wl,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]), linewidth=0.6)
    plt.ylabel(r'$I_{TPPL}\, /\, a.u.$')
    plt.xlabel(r'$\lambda\, /\, nm$')
    plt.legend(['Experiment', 'Lorentz-Fit'])
    plt.tight_layout()
    plt.savefig(savedir + str(index) + ".jpg", dpi=400)
    plt.close()
    img_fit[i, :] = two_lorentz(wl,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])
    wl1[i] = popt[3]
    wl2[i] = popt[4]
    wl1_ind[i] = np.argmin( np.abs( wl - popt[3] ) )
    wl2_ind[i] = np.argmin(np.abs(wl - popt[4]))

# newfig(0.9)
# plt.imshow(img_fit, aspect='auto', cmap=plt.get_cmap("viridis"), extent=[wl.min(), wl.max(), sizes.min(), sizes.max()])
# plt.ylabel(r'$Size / nm$')
# plt.xlabel(r'$\lambda\, /\, nm$')
# plt.tight_layout()
# plt.savefig(savedir + "image_fitted.pdf")
# plt.savefig(savedir + "image_fitted.png", dpi=400)
# # plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
# plt.close()


fig, ax1 = newfig(0.9)
ax1.scatter(sizes,wl1,color='C0')
ax1.set_ylabel(r'$Shift\, of\, Broad Mode\, /\, nm$',color='C0')
ax1.set_ylim([wl1.mean()-(wl2.max()-wl2.min())/2,wl1.mean()+(wl2.max()-wl2.min())/2 ])

ax2 = ax1.twinx()
ax2.scatter(sizes,wl2,s=20,marker="+",color='C1')
ax2.set_ylabel(r'$Shift\, of \,Sharp Mode\, /\, nm$',color='C1')

ax1.set_xlabel(r'$Size / nm$')

plt.tight_layout()
plt.savefig(savedir + "shifts.pdf")
os.system("convert -density 1200  "+savedir+"shifts.pdf -quality 100 -resample 300 "+savedir+"shifts.jpg")
#plt.savefig(savedir + "shifts.jpg", dpi=400)
plt.close()

fig, ax1 = newfig(0.9)
ax1.scatter(sizes,wl1-wl1[0],color='C0')
ax1.scatter(sizes,wl2-wl2[0],marker="+",color='C1')
ax1.set_ylabel(r'$rel.\, Spectral\, Shift\, /\, nm$')
ax1.set_xlabel(r'$Size / nm$')
plt.legend(['Broad Mode','Sharp Mode'])
plt.tight_layout()
plt.savefig(savedir + "shifts_rel.pdf")
os.system("convert -density 1200  "+savedir+"shifts_rel.pdf -quality 100 -resample 300 "+savedir+"shifts_rel.jpg")
#plt.savefig(savedir + "shifts_rel.jpg", dpi=400)
plt.close()


# x = wl
# y = df[index]
# #p0 = [0, 0.15,0.1,0.1, 550, 613, 630, 150, 15, 15]
# p0 = [0, 0.15,0.1,0.1, 550, 613, 630, 150, 10, 30]
#
#
# plt.plot(wl,df[index])
# plt.plot(wl,three_lorentz(x,p0[0],p0[1],p0[2],p0[3],p0[4],p0[5],p0[6],p0[7],p0[8],p0[9]))
# plt.show()
#
# for i,index in enumerate(sizes_ind):
#     x = wl
#     y = df[index]
#
#     popt, pcov = curve_fit(three_lorentz, x, y, p0)
#     p0 = popt
#
#     plt.plot(wl,df[index])
#     plt.plot(wl,three_lorentz(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9]))
#     plt.savefig(savedir + str(index) + ".png", dpi=400)
#     plt.close()

# popt, pcov = curve_fit(three_gauss, x, y, p0)
# print(popt)

# plt.plot(wl,df[index])
# plt.plot(wl,three_lorentz(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9]))
# plt.show()

#
# img = np.zeros((len(sizes_ind), len(wl)))
# for i,index in enumerate(sizes_ind):
#     img[i, :] = savgol_filter(df[index],21,0)
#
# # img = ndimage.median_filter(img, 5)
#
# newfig(0.9)
# plt.imshow(img, aspect='auto', cmap=plt.get_cmap("viridis"), extent=[wl.min(), wl.max(), sizes.min(), sizes.max()])
# plt.ylabel(r'$Size / nm$')
# plt.xlabel(r'$\lambda\, /\, nm$')
# plt.tight_layout()
# plt.savefig(savedir + "image_filtered.pdf")
# plt.savefig(savedir + "image_filtered.png", dpi=400)
# # plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
# plt.close()
#
img = np.zeros((len(sizes_ind), len(wl)))
for i,index in enumerate(sizes_ind):
    img[i, :] = df[index]
#
# newfig(0.9)
# plt.imshow(img, aspect='auto', cmap=plt.get_cmap("viridis"), extent=[wl.min(), wl.max(), sizes.min(), sizes.max()])
# plt.ylabel(r'$Size / nm$')
# plt.xlabel(r'$\lambda\, /\, nm$')
# plt.tight_layout()
# plt.savefig(savedir + "image.pdf")
# plt.savefig(savedir + "image.png", dpi=400)
# # plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
# plt.close()

# cmap = plt.get_cmap('Dark2')
# colors = [cmap(i) for i in np.linspace(0, 1, len(sizes))]
#
# for i,index in enumerate(sizes_ind):
#     y = df[index]/img.max()
#     y_filt = savgol_filter(y,41,0)
#
#     p1 = plt.plot(wl,y+i*0.3, color=colors[i], linewidth=0.5,alpha = 0.5)
#     #plt.setp(p1)
#     p2 = plt.plot(wl,y_filt+i*0.3, color=colors[i], linewidth=0.5,alpha = 1.0)
#     #plt.setp(p2)
#
#     #plt.scatter(peak_wl,filtered[np.argmax(filtered)]+i*0.3,s=20,marker="x")
#     #plt.text(wl,y_filt*1.1+i*0.3,str(index)+'nm')
#
#
# plt.ylabel(r'$I_{TPPL}\, /\, a.u.$')
# plt.xlabel(r'$\lambda\, /\, nm$')
# plt.tight_layout()
# # plt.ylim([0, (len(pics)+1)*0.3*1.1])
# plt.savefig(savedir + "dimer_waterfall.pdf")
# plt.savefig(savedir + "dimer_waterfall.png", dpi= 400)
# plt.close()

cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 1, len(sizes)+3)]

plt.clf()
fig = plt.figure(figsize=[2.0,5.5])
ax = fig.add_subplot(111)

ax.axvline(500,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(550,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(600,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(650,color='black', linestyle='--', linewidth=0.5,alpha=0.5)


for i,index in enumerate(sizes_ind):
    x = wl[mask]
    y = img[i, mask]
    y_fit = img_fit[i, :]

    #p1 = ax.plot(x,y+i*0.3, color=colors[i], linewidth=0.5,alpha = 0.7,marker='.',linestyle='',markeredgewidth=0,markersize=2.0)
    #plt.setp(p1)

    p0 = ax.plot(wl[mask1],img[i, mask1]+i*0.3, color=colors[i], linewidth=0.5,alpha = 0.9,linestyle='-')
    p1 = ax.plot(wl[mask2], img[i, mask2] + i * 0.3, color=colors[i], linewidth=0.5, alpha=0.9, linestyle='-')

    p2 = ax.plot(wl,y_fit+i*0.3, color='black', linewidth=0.7,alpha = 1.0)
    #plt.setp(p2)

    #plt.scatter(peak_wl,filtered[np.argmax(filtered)]+i*0.3,s=20,marker="x")
    #plt.text(wl,y_filt*1.1+i*0.3,str(index)+'nm')


#ax.set_ylabel(r'$I_{TPPL}\, /\, a.u.$')
#ax.set_xlabel(r'$\lambda\, /\, nm$')
ax.set_ylabel(r'Normalized MPL spectra')
ax.set_xlabel(r'Wavelength (nm)')
ax.set_ylim([img[0, :].min(), img[-1, :].max()+0.3*(len(sizes)-1)])


curve1x = np.zeros(len(wl1_ind))
curve1y = np.zeros(len(wl1_ind))
for i in range(len(wl1_ind)):
    curve1x[i] = wl[wl1_ind[i]]
    curve1y[i] = img_fit[i,wl1_ind[i]]+i*0.3

ax.plot(curve1x,curve1y,color='blue', linewidth=1.3, linestyle='--')

curve2x = np.zeros(len(wl2_ind))
curve2y = np.zeros(len(wl2_ind))
yticks = []
for i in range(len(wl2_ind)):
    curve2x[i] = wl[wl2_ind[i]]
    curve2y[i] = img_fit[i, wl2_ind[i]]+i*0.3
    yticks.append(img_fit[i, -1]+i*0.3)

ax.plot(curve2x, curve2y,color='lime', linewidth=1.3, linestyle='--')


ax.tick_params(axis='y', which='both',left='off',right='off', labelleft='off', labelright='on')
ax.set_yticks(yticks)
labels = []
for index in sizes_ind:
    labels.append('d='+str(index))
ax.set_yticklabels(labels)

ax.set_xticks([500,600])
ax.set_xticklabels(['500','600'])

plt.tight_layout()
# plt.ylim([0, (len(pics)+1)*0.3*1.1])
plt.savefig(savedir + "waterfall_fit.pdf")
os.system("convert -density 1200  "+savedir+"waterfall_fit.pdf -quality 100 -resample 600 "+savedir+"waterfall_fit.jpg")
#plt.savefig(savedir + "waterfall_fit.jpg", dpi= 600)
plt.close()


df_fit = pd.DataFrame({
                        sizes_ind[0] : img_fit[0,:],
                        sizes_ind[1] : img_fit[1,:],
                        sizes_ind[2] : img_fit[2,:],
                        sizes_ind[3] : img_fit[3,:],
                        sizes_ind[4] : img_fit[4,:],
                        sizes_ind[5] : img_fit[5, :],
                        sizes_ind[6] : img_fit[6, :],
                        sizes_ind[7] : img_fit[7, :],
                        'Wavelength' : wl,
                        })

df_peaks = pd.DataFrame({
                            'broad mode': wl1,
                            'sharp mode': wl2,
                            'Sizes': sizes,
                        })

writer = pd.ExcelWriter(path+'FittedData_TPL.xlsx')
df_fit.to_excel(writer,'fits')
df_peaks.to_excel(writer,'peaks')
writer.save()


plt.clf()
fig = plt.figure(figsize=[2.0,5.5])
ax = fig.add_subplot(111)

ax.axvline(500,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(550,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(600,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(650,color='black', linestyle='--', linewidth=0.5,alpha=0.5)


for i,index in enumerate(sizes_ind):
    x = wl[mask]
    y = img[i, mask]
    y_fit = img_fit[i, :]

    #p1 = ax.plot(x,y+i*0.3, color=colors[i], linewidth=0.5,alpha = 0.7,marker='.',linestyle='',markeredgewidth=0,markersize=2.0)
    #plt.setp(p1)

    p0 = ax.plot(wl[mask1],img[i, mask1]+i*0.3, color=colors[i], linewidth=0.5,alpha = 0.9,linestyle='-')
    p1 = ax.plot(wl[mask2], img[i, mask2] + i * 0.3, color=colors[i], linewidth=0.5, alpha=0.9, linestyle='-')

    #p2 = ax.plot(wl,y_fit+i*0.3, color='black', linewidth=0.7,alpha = 1.0)
    #plt.setp(p2)

    #plt.scatter(peak_wl,filtered[np.argmax(filtered)]+i*0.3,s=20,marker="x")
    #plt.text(wl,y_filt*1.1+i*0.3,str(index)+'nm')


#ax.set_ylabel(r'$I_{TPPL}\, /\, a.u.$')
#ax.set_xlabel(r'$\lambda\, /\, nm$')
ax.set_ylabel(r'Normalized MPL spectra')
ax.set_xlabel(r'Wavelength (nm)')
ax.set_ylim([img[0, :].min(), img[-1, :].max()+0.3*(len(sizes)-1)])


curve1x = np.zeros(len(wl1_ind))
curve1y = np.zeros(len(wl1_ind))
for i in range(len(wl1_ind)):
    curve1x[i] = wl[wl1_ind[i]]
    curve1y[i] = img_fit[i,wl1_ind[i]]+i*0.3

ax.plot(curve1x,curve1y,color='blue', linewidth=1.3, linestyle='--')

curve2x = np.zeros(len(wl2_ind))
curve2y = np.zeros(len(wl2_ind))
yticks = []
for i in range(len(wl2_ind)):
    curve2x[i] = wl[wl2_ind[i]]
    curve2y[i] = img_fit[i, wl2_ind[i]]+i*0.3
    yticks.append(img_fit[i, -1]+i*0.3)

ax.plot(curve2x, curve2y,color='lime', linewidth=1.3, linestyle='--')


ax.tick_params(axis='y', which='both',left='off',right='off', labelleft='off', labelright='on')
ax.set_yticks(yticks)
labels = []
for index in sizes_ind:
    labels.append('d='+str(index))
ax.set_yticklabels(labels)

ax.set_xticks([500,600])
ax.set_xticklabels(['500','600'])

plt.tight_layout()
# plt.ylim([0, (len(pics)+1)*0.3*1.1])
plt.savefig(savedir + "waterfall.pdf")
os.system("convert -density 1200  "+savedir+"waterfall.pdf -quality 100 -resample 600 "+savedir+"waterfall.jpg")
#plt.savefig(savedir + "waterfall.jpg", dpi= 600)
plt.close()




cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 1, len(sizes)+3)]

plt.clf()
fig = plt.figure(figsize=[2.0,5.5])
ax = fig.add_subplot(111)

ax.axvline(500,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(550,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(600,color='black', linestyle='--', linewidth=0.5,alpha=0.5)
ax.axvline(650,color='black', linestyle='--', linewidth=0.5,alpha=0.5)


y_pos = sizes/sizes.max()
print(y_pos)

img /= len(y_pos)
img_fit /= len(y_pos)

y_max = 0
y_min = np.inf

for i,index in enumerate(sizes_ind):
    x = wl[mask]
    y = img[i, mask]
    y_fit = img_fit[i, :]

    #p1 = ax.plot(x,y+i*0.3, color=colors[i], linewidth=0.5,alpha = 0.7,marker='.',linestyle='',markeredgewidth=0,markersize=2.0)
    #plt.setp(p1)

    p0 = ax.plot(wl[mask1],img[i, mask1]+y_pos[i], color=colors[i], linewidth=0.5,alpha = 0.9,linestyle='-')
    p1 = ax.plot(wl[mask2], img[i, mask2]+y_pos[i], color=colors[i], linewidth=0.5, alpha=0.9, linestyle='-')

    p2 = ax.plot(wl,y_fit+y_pos[i], color='black', linewidth=0.7,alpha = 1.0)
    #plt.setp(p2)

    #plt.scatter(peak_wl,filtered[np.argmax(filtered)]+i*0.3,s=20,marker="x")
    #plt.text(wl,y_filt*1.1+i*0.3,str(index)+'nm')
    y_max = np.max([y_max,img[i, mask1].max()+y_pos[i]])
    y_min = np.min([y_min, img[i, mask1].min() + y_pos[i]])

#ax.set_ylabel(r'$I_{TPPL}\, /\, a.u.$')
#ax.set_xlabel(r'$\lambda\, /\, nm$')
ax.set_ylabel(r'Normalized MPL spectra', family="sans-serif")
ax.set_xlabel(r'Wavelength (nm)', family="sans-serif")
#ax.set_ylim([img[0, :].min(), img[-1, :].max()+1])
ax.set_ylim([y_min, y_max])


curve1x = np.zeros(len(wl1_ind))
curve1y = np.zeros(len(wl1_ind))
for i in range(len(wl1_ind)):
    curve1x[i] = wl[wl1_ind[i]]
    curve1y[i] = img_fit[i,wl1_ind[i]]+y_pos[i]

ax.plot(curve1x,curve1y,color='blue', linewidth=1.3, linestyle='--')

curve2x = np.zeros(len(wl2_ind))
curve2y = np.zeros(len(wl2_ind))
yticks = []
for i in range(len(wl2_ind)):
    curve2x[i] = wl[wl2_ind[i]]
    curve2y[i] = img_fit[i, wl2_ind[i]]+y_pos[i]
    yticks.append(img_fit[i, -1]+y_pos[i])

ax.plot(curve2x, curve2y,color='lime', linewidth=1.3, linestyle='--')


ax.tick_params(axis='y', which='both',left='off',right='off', labelleft='off', labelright='on')
ax.set_yticks(yticks)
labels = []
for index in sizes_ind:
    labels.append('d='+str(index))
ax.set_yticklabels(labels)

ax.set_xticks([500,600])
ax.set_xticklabels(['500','600'])

plt.tight_layout()
# plt.ylim([0, (len(pics)+1)*0.3*1.1])
plt.savefig(savedir + "waterfall_fit_linear.pdf")
os.system("convert -density 1200  "+savedir+"waterfall_fit_linear.pdf -quality 100 -resample 600 "+savedir+"waterfall_fit_linear.jpg")
#plt.savefig(savedir + "waterfall_fit_linear.jpg", dpi= 600)
plt.close()
