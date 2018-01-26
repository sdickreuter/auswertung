import os

import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import scipy.optimize as opt
from scipy.optimize import curve_fit, basinhopping
import scipy.sparse as sparse
from scipy.special import *
from plotsettings import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
import re
import scipy.signal as signal
import peakutils as pu

def lorentz(x, amplitude, x0, sigma):
    g = (amplitude*2/(np.pi*sigma))/(1+4*np.square((x-x0)/sigma))
    return g.ravel()

def gauss(x, amplitude, x0, sigma):
    g = amplitude/sigma * np.sqrt(4*np.log(2)/np.pi)*np.exp(-4*np.log(2)*np.square((x-x0)/sigma))
    return g.ravel()

# https://www.webpages.uidaho.edu/brauns/vibspect1.pdf
def asymvoigt(x, amplitude, x0, sigma, a , f):
    sigma = 2 * sigma/(1 + np.exp(a*(x-x0)) )
    g = f*lorentz(x,amplitude,x0,sigma)+(1-f)*gauss(x,amplitude,x0,sigma)
    return g.ravel()

def fit_fun(x, amp, x0, sigma,a,f):
    return asymvoigt(x, amp, x0, sigma,a,f)

path = '/home/sei/Raman/Fatima3/'
savedir = path + 'plots/'

peak_pos = [1085,1590]
search_width = 100  # cm^-1

try:
    os.mkdir(savedir)
except:
    pass


files = []
for file in os.listdir(path):
    if re.search(r"\.(txt)$", file) is not None:
        files.append(file)

print(files)


#file = files[0]
k_max = np.zeros((len(files),len(peak_pos)))
c_max = np.zeros((len(files),len(peak_pos)))
labels = np.array([])

for i,file in enumerate(files):
    print(file)

    k, counts = np.loadtxt(path + file, unpack=True)

    counts = signal.savgol_filter(counts, 31, 1, mode='interp')

    base = pu.baseline(counts, 11, max_it=10000, tol=0.00001)
    counts -= base

    #newfig(0.9)
    plt.plot(k, counts, linewidth=1)
    # plt.plot(k, bl, linewidth=1)
    # plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.6)
    plt.ylabel(r'$I_{\nu}\, /\, counts$')
    plt.xlabel(r'$wavenumber\, /\, cm^{-1}$')
    # plt.xlim((minwl, maxwl))
    # plt.plot(wl, counts)
    plt.tight_layout()
    #plt.show()
    plt.savefig(savedir + file[:-4] + ".pdf", dpi=300)
    plt.close()

    for j,peak in enumerate(peak_pos):
        mask = (k <= peak + search_width) & (k >= peak - search_width)
        c1 = counts[mask]
        k1 = k[mask]

        max_ind = np.argmax(c1)
        k_max[i,j] = k1[max_ind]
        c_max[i,j] = c1[max_ind]

    labels = np.append(labels,file[:-6])


print(c_max)



sort = np.argsort(labels)
labels = labels[sort]
k_max = k_max[sort,:]
c_max = c_max[sort,:]

print(labels)
label = np.unique(labels)
print(label)

for l in label:
    mask = labels == l
    plt.scatter(k_max[mask], c_max[mask])

plt.savefig(path + "scatter.pdf", dpi=300)
plt.close()

mean = np.zeros((len(label),len(peak_pos)))
err = np.zeros((len(label),len(peak_pos)))

for i,l in enumerate(label):
    mask = labels == l

    for j in range(len(peak_pos)):
        mean[i,j] = np.mean(c_max[mask,j])
        err[i,j] = np.std(c_max[mask,j])

print(mean)

print(mean[:,0].ravel())
print(np.arange(0,mean.shape[0],1))
for i in range(mean.shape[1]):
    plt.bar(np.arange(0,mean.shape[0],1)*mean.shape[1]+(i+1),mean[:,i].ravel(),yerr=err[:,i].ravel())

plt.xticks((np.arange(0,mean.shape[0],1)*mean.shape[1]+(mean.shape[1]+1)/2), label)

plt.savefig(path + "bar.pdf", dpi=300)
plt.close()


print('-> Writing measured values to file')
with open(path + "raman.csv", 'w') as f:
    f.write("label,")
    for j in range(mean.shape[1]):
        f.write("mean"+str(peak_pos[j])+",err"+str(peak_pos[j])+",")

    f.write("\r\n")


    for i in range(len(label)):
        f.write( label[i] + ",")
        for j in range(mean.shape[1]):
            f.write( str(mean[i,j])+ "," + str(err[i,j])+"," )


        f.write("\r\n")



mean = np.zeros((len(label),len(counts)))
err = np.zeros((len(label),len(counts)))
for i, l in enumerate(label):

    buf = []
    for j,file in enumerate(files):
        if file[:-6] == l:

            k, counts = np.loadtxt(path + file, unpack=True)

            #counts = signal.savgol_filter(counts, 31, 1, mode='interp')

            #base = pu.baseline(counts, 11, max_it=10000, tol=0.00001)
            #counts -= base

            buf.append(counts)

    buf = np.array(buf)
    print(buf.shape)
    mean[i, :] = np.mean(buf,axis=0)
    err[i, :] = np.std(buf,axis=0)


fig, ax = newfig(0.9)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


legend = ["A: 30 min","B: 30 min","C: 90 min","D: 90 min"]
print(label)
print(legend)

for i, l in enumerate(label):
    poly = np.array((k,mean[i,:]+err[i,:]+1000*i))
    poly = np.hstack((poly,np.fliplr(np.array((k, mean[i,:] - err[i,:]+1000*i)))))
    poly = poly.T
    ax.add_patch(Polygon(poly, closed=True,fill=True,alpha = 0.3,facecolor=colors[i]))
    #plt.plot(wl, mean_spec, linewidth=0.8)
    plt.plot(k,mean[i,:]+1000*i, linewidth=0.8)

plt.ylabel(r'$I_{\nu}\, /\, counts$')
plt.xlabel(r'$\Delta\widetilde{\nu}\, /\, cm^{-1}$')
plt.legend(legend)
plt.tight_layout()
plt.savefig(path + "overview.pdf", dpi=300)
plt.close()


    # width = 100
    # max_ind = np.argmax(counts)
    # indices = np.arange(0, len(k), 1)
    # mask = (indices <= max_ind + width) & (indices >= max_ind - width)
    # # inds = np.arange(max_ind-width,max_ind+width,1)
    # k1 = k[mask]
    # counts1 = counts[mask]



    # def err_fun(p):
    #     fit = fit_fun(k1, *p)
    #     diff = np.abs(counts1 - fit)
    #     return np.sum(diff)
    #
    # #def fit_fun(x, amp, x0, sigma,a,f,b,c):
    # b = 0# ( np.mean(counts1[20:])-np.mean(counts1[:-20]) )/( np.mean(k1[20:])-np.mean(k1[:-20]) )
    # c = 0#np.mean(k1[20:])
    # start = [counts[max_ind]*3,k[max_ind],150,0.01,0.1]
    # upper = [counts[max_ind]*10, k[max_ind]+width, 500, 1,1]
    # lower = [                  0, k[max_ind]-width,    10, 0,0]
    # bnds = []
    # for i in range(len(upper)):
    #     bnds.append((lower[i], upper[i]))
    #
    # #minimizer_kwargs = {"method": "SLSQP","bounds": bnds,"tol":1e-10}
    # #res = basinhopping(err_fun, start, minimizer_kwargs=minimizer_kwargs, niter=1000,disp=False)
    # res = opt.minimize(err_fun, start, method='SLSQP', options={'disp': True, 'maxiter': 10000},tol=1e-10)
    # #res = opt.minimize(err_fun, start, method='L-BFGS-B', options={'disp': True, 'maxiter': 5000})
    # #res = opt.minimize(err_fun, start, method='Nelder-Mead', options={'disp': True, 'maxiter': 5000})
    #
    # popt = res.x
    #
    # print(popt)
    # plt.plot(k1, counts1, linewidth=1)
    # plt.plot(k1, fit_fun(k1,popt[0],popt[1],popt[2],popt[3],popt[4]), linewidth=1)
    # #plt.plot(k1, popt[5]*k1+popt[6])
    # plt.ylabel(r'$I_{\nu}\, /\, counts$')
    # plt.xlabel(r'$wavenumber\, /\, cm^{-1}$')
    # plt.savefig(savedir + file[:-4] + "fit.pdf", dpi=300)
    # #plt.show()
    # plt.close()
    #
    # fit = fit_fun(k1,popt[0],popt[1],popt[2],popt[3],popt[4])
    # print(np.max(fit))

