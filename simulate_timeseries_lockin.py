import os
import re
from scipy.interpolate import interp1d

import numpy as np
from scipy.optimize import minimize, curve_fit, basinhopping
from scipy.signal import savgol_filter
from scipy import stats
#from plotsettings import *

import matplotlib.pyplot as plt
from numba import jit

#path = '/home/sei/Spektren/p42_lockintest/'
#path = '/home/sei/Spektren/p57m_did6_zeiss/'
#path = '/home/sei/data/ts4/'
path = '/home/sei/data/A1_fsl/'
#path = '/home/sei/data/tric6c3/'


maxwl = 1000
minwl = 400

savedir = "./plots/"

@jit(nopython=True)
def lockin_filter(signal, reference):
    width = signal.shape[0]
    y = np.zeros(width)
    for ind in range(width):
        y[ind] = np.sum(signal[ind,:] * reference) / len(reference)

    return y

@jit()
def interpolate_data(x,x_new,signal):
    width = signal.shape[0]
    res = np.zeros((signal.shape[0],len(x_new)))
    for ind in range(width):
        spline = interp1d(x, signal[ind, :], kind='cubic')
        res[ind, :] = spline(x_new)
    return res


def lorentz(x, amplitude, xo, sigma):
    xo = float(xo)
    g = amplitude * np.power(sigma / 2, 2.) / (np.power(sigma / 2, 2.) + np.power(x - xo, 2.))
    return g.ravel()

def gauss(x, amplitude, xo, fwhm):
    sigma = fwhm / 2.3548
    xo = float(xo)
    g = amplitude * np.exp(-np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))
    return g.ravel()

wl = np.linspace(minwl,maxwl,2000)
mask = np.repeat(True,len(wl))

spec1 = lorentz(wl,1,540,100)
spec2 = (lorentz(wl,2,600,120)+lorentz(wl,0.5,450,90))
lamp = gauss(wl,2,600,300)

plt.plot(wl,spec1)
plt.plot(wl,spec2)
plt.plot(wl,lamp)
plt.show()

n = 1000
t2 = np.linspace(0,900,n)
ts = np.zeros((len(wl),n))

phase = 0# np.pi/5
f = 0.005

ref = np.sin(2 * np.pi * t2 * f/2 + phase)
ref2 = np.sin(2 * np.pi * t2 * f/2  + phase - np.pi)
for i in range(len(wl)):
    ts[i,:] += (ref+0.1) * spec1[i] + np.random.rand(n)*0.3
    ts[i,:] += (ref2+0.1) * spec2[i] + np.random.rand(n)*0.3
    ts[i,:] *= (lamp[i] + np.random.rand(n)*0.3)

ts += 1

plt.imshow(ts.T, extent=(wl.min(), wl.max(), t2.max(), t2.min()), aspect=1,cmap=plt.get_cmap("seismic"))
plt.xlabel("Wavelength / nm")
plt.ylabel("t / s")
plt.show()

width = ts.shape[0]



@jit()
def func(x,a,f,p,c):
    return a*np.sin(2 * np.pi * x * f + p) + c

res = np.zeros(width)
res_fit = np.zeros(width)
res_phase = np.zeros(width)
res_fit_phase = np.zeros(width)
freqs = np.zeros(width)
#for ind in np.arange(400,500):
for ind in range(width):
    print(ind)
    fft = np.fft.rfft(ts[ind,:],norm="ortho")
    d = np.absolute(fft)
    p = np.angle(fft)
    res[ind] = d[1:].max()
    res_phase[ind] = p[d[1:].argmax()+1]

    f_buf = np.fft.rfftfreq(t2.shape[0], d=t2[1] - t2[0])
    freqs[ind] = f_buf[d[1:].argmax()+1]

    # if freqs[ind] <= 0:
    #     initial_guess = [y.max()-y.min(),0.003,np.pi,np.mean(y)]
    # else:
    #     initial_guess = [y.max() - y.min(), freqs[ind], 0, np.mean(y)]
    #
    # bounds = ( [initial_guess[0] * 0.5, 0, 0, y.min()], [initial_guess[0] * 1.5, 1, 2*np.pi, y.max()])
    # #bounds = [(initial_guess[0] * 0.7, initial_guess[0] * 1.2), (0, 1), (0, 2*np.pi), (y.min(), y.max())]
    # #optimize_func = lambda x: np.sum(np.square(np.subtract(func(t2,x[0],x[1],x[2],x[3]),y)))
    # #minimizer_kwargs = {"method": "L-BFGS-B", "bounds" : bounds}
    # try:
    #     #opt = basinhopping(optimize_func, initial_guess, minimizer_kwargs=minimizer_kwargs)
    #     #opt = minimize(optimize_func, initial_guess, method="SLSQP")#,bounds=bounds)
    #     opt, pcov = curve_fit(func, t2, y, p0=initial_guess)#,bounds=bounds)
    #     #opt = opt.x
    #     res_fit[ind] = opt[0]
    #     res_fit_phase[ind] = opt[2]
    # except RuntimeError:
    #     print("fit error")

mask2 = wl > 0#(wl >= 400) & (wl <= 600)
freqs = freqs[mask2]
freqs = freqs[np.argwhere(res[mask2] > np.max(res[mask2])*0.9)]
freqs = freqs[:,0]
freqs = freqs[np.argwhere(freqs > 0)]

res_phase += np.pi
phase = res_phase[np.argmax(res)]

f = freqs[0]
print("freq1: "+str(f))

maxind = np.argmax(res)
y = ts[maxind, :]

fft = np.fft.rfft(y, norm="ortho")
d = np.absolute(fft)
p = np.angle(fft)
f_buf = np.fft.rfftfreq(t2.shape[0], d=t2[1] - t2[0])
print("freq fft: "+str(f_buf[d[1:].argmax()]))


initial_guess = [y.max() - y.min(), f, 0, np.mean(y)]
bounds = ([initial_guess[0] * 0.5, 0, 0, y.min()], [initial_guess[0] * 1.5, 1, 2 * np.pi, y.max()])
#optimize_func = lambda x: np.sum(np.square(np.subtract(func(t2, x[0], x[1], x[2], x[3]), y)))
#minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
#popt = basinhopping(optimize_func, initial_guess, minimizer_kwargs=minimizer_kwargs)
#popt = minimize(optimize_func, initial_guess, method="SLSQP")#, bounds=bounds)
#popt = popt.x
popt, pcov = curve_fit(func, t2, y, p0=initial_guess)#,bounds=bounds)
#print(popt)

fig = plt.figure()
plt.plot(t2,y)
plt.plot(t2,func(t2,popt[0],popt[1],popt[2],popt[3]))
#plt.plot(t2,func(t2,popt[0],f,phase,popt[3]))
plt.savefig(savedir + "fit.png",dpi=300)
plt.close()

f = popt[1]
phase = popt[2]
print("freq2: "+str(f))





#plt.plot(freqs)
#plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
#ax.axhline(0)
res_phase = res_phase * 180/np.pi
p1 = ax.plot(wl[mask], res[mask], alpha=0.5)  # /lamp[mask])
res2 = savgol_filter(res, 71, 1)
p2 = ax.plot(wl[mask], res2[mask])  # /lamp[mask])
ax.tick_params('y', colors=p2[0].get_color())

ax2 = ax.twinx()

next(ax2._get_lines.prop_cycler)
next(ax2._get_lines.prop_cycler)

p3 = ax2.plot(wl[mask], res_phase[mask], alpha=0.5)  # /lamp[mask])
ax2.tick_params('y', colors=p3[0].get_color())

ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
ax.patch.set_visible(False)  # hide the 'canvas'

plt.savefig(savedir + "lockin_fft.png",dpi=300)
plt.close()

ref = np.sin(2 * np.pi * t2 * f)
ref2 = np.sin(2 * np.pi * t2 * f + np.pi)
#ref = -sawtooth(2 * np.pi * x * f * 2, 0.5)
#ref2 = -sawtooth(2 * np.pi * x * f * 2 + np.pi / 2, 0.5)

res = np.zeros(width)
res_phase = np.zeros(width)
res_amp = np.zeros(width)
x2 = np.zeros(width)
y2 = np.zeros(width)

x2 = lockin_filter(ts,ref)
y2 = lockin_filter(ts,ref2)

# res_phase = savgol_filter(res_phase,3,1)
res_amp = np.abs(x2 + 1j * y2)
res_phase = np.angle(x2 + 1j * y2)
maxind = np.argmax(savgol_filter(res_amp, 71, 1))
phase = res_phase[maxind]
print("phase1: "+str(phase))

phases = np.linspace(0,2*np.pi,3600)
x1 = np.zeros(len(phases))
for i in range(len(phases)):
    ref1 = np.sin(2 * np.pi * t2 * f + phases[i])
    x1[i] = lockin_filter(np.matrix(ts[maxind,:]),ref1)

# plt.plot(phases,x1)
# plt.show()
phase = phases[np.argmax(x1)]
print("phase1: "+str(phase))

res_phase = np.angle( (x2*np.cos(phase)-y2*np.sin(phase)) + 1j * ( x2*np.sin(phase) + y2*np.cos(phase) ))
#res_phase += phase
#res_phase += np.pi


# print("Max Amp Phase"+str(res_phase[np.argmax(res_amp)]))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t2, ref / np.max(ref) + 0)
ax.plot(t2, ref2 / np.max(ref2) + 1)

indices = [np.argmin(np.abs(wl - 500)), np.argmin(np.abs(wl - 600)), np.argmin(np.abs(wl - 700)),
           np.argmin(np.abs(wl - 810))]
for i, ind in enumerate(indices):
    buf = ts[ind,:]
    buf = buf - np.min(buf)
    ax.plot(t2, buf / np.max(buf) + i + 2)
    #ax.plot(x, ref / ref.max() + i)

plt.savefig(savedir + "traces.png",dpi=300)
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)
indices = [np.argmin(np.abs(wl - 500)), np.argmin(np.abs(wl - 600)), np.argmin(np.abs(wl - 700)),
           np.argmin(np.abs(wl - 810))]
for i, ind in enumerate(indices):

    fft = np.fft.rfft(ts[ind,:])
    d = np.absolute(fft)
    p = np.abs(np.angle(fft))
    f_buf = np.fft.rfftfreq(t2.shape[0])
    blub = int(len(f_buf)/4)
    ax.plot(f_buf[1:blub], d[1:blub] / d[1:blub].max() + i)
    ax.plot(f_buf[1:blub], p[1:blub] / p[1:blub].max() + i, alpha=0.5)

# plt.axvline(x=f)
# plt.axvline(x=f * 2)
plt.savefig(savedir + "fft.png",dpi=300)
plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# res_phase = np.abs(res_phase)
# ax.plot(wl[mask], res_phase[mask] / res_phase[mask].max(), alpha=0.5)  # /lamp[mask])
# ax.plot(wl[mask], res[mask] / res[mask].max(), alpha=0.5)  # /lamp[mask])
# res2 = savgol_filter(res, 71, 1)
# ax.plot(wl[mask], res2[mask] / res[mask].max())  # /lamp[mask])
# plt.savefig(savedir + "lockin.png")
# plt.close()
#print(wl[np.argmax(res2)])

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.axhline(0)
res_phase = res_phase * 180/np.pi
p1 = ax.plot(wl[mask], res_amp[mask], alpha=0.5)  # /lamp[mask])
res2 = savgol_filter(res_amp, 71, 1)
p2 = ax.plot(wl[mask], res2[mask])  # /lamp[mask])
ax.tick_params('y', colors=p2[0].get_color())
ax2 = ax.twinx()
next(ax2._get_lines.prop_cycler)
next(ax2._get_lines.prop_cycler)
p3 = ax2.plot(wl[mask], res_phase[mask], alpha=0.5)  # /lamp[mask])
ax2.tick_params('y', colors=p3[0].get_color())

ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
ax.patch.set_visible(False)  # hide the 'canvas'

plt.savefig(savedir + "lockin_nopsd.png",dpi=300)
plt.close()

plt.plot(wl, res_amp)
plt.plot(wl,spec1)
plt.plot(wl,spec2)
plt.show()


ref = np.sin(2 * np.pi * t2 * f + phase)
ref2 = np.sin(2 * np.pi * t2 * f  + phase - np.pi)

x2 = lockin_filter(ts,ref)
x3 = lockin_filter(ts,ref2)

#x2[x2<0] = 0
#x3[x3<0] = 0

fig = plt.figure()
ax = fig.add_subplot(111)

p1 = ax.plot(wl[mask], x2[mask], alpha=0.5)  # /lamp[mask])
res2 = savgol_filter(x2, 71, 1)
p2 = ax.plot(wl[mask], res2[mask])  # /lamp[mask])
ax.tick_params('y', colors=p2[0].get_color())

ax2 = ax.twinx()
p3 = ax.plot(wl[mask], x3[mask], alpha=0.5)  # /lamp[mask])
res2 = savgol_filter(x3, 71, 1)
p4 = ax.plot(wl[mask], res2[mask])  # /lamp[mask])
ax.tick_params('y', colors=p2[0].get_color())

ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
ax.patch.set_visible(False)  # hide the 'canvas'

plt.savefig(savedir + "lockin_psd.png", dpi=300)
plt.close()


fig = plt.figure()
ax = fig.add_subplot(111)
p1 = ax.plot(wl[mask], x2[mask]*x3[mask], alpha=0.5)  # /lamp[mask]
res2 = savgol_filter(x2*x3, 71, 1)
p2 = ax.plot(wl[mask], res2[mask])  # /lamp[mask])
plt.savefig(savedir + "lockin_psd_mult.png", dpi=300)
plt.close()


ref1 = np.sin(2 * np.pi * t2 * f + phase )
ref2 = np.sin(2 * np.pi * t2 * f + phase + (2*np.pi)/3)
ref3 = np.sin(2 * np.pi * t2 * f + phase - (2*np.pi)/3)

x1 = lockin_filter(ts,ref1)
x2 = lockin_filter(ts,ref2)
x3 = lockin_filter(ts,ref3)


#x1[x1<0] = 0
#x2[x2<0] = 0
#x3[x3<0] = 0

fig = plt.figure()
ax = fig.add_subplot(111)

p1 = ax.plot(wl[mask], savgol_filter(np.abs(x1[mask]), 71, 1), alpha=1)  # /lamp[mask])
p2 = ax.plot(wl[mask], savgol_filter(np.abs(x2[mask]), 71, 1), alpha=1)  # /lamp[mask])
p3 = ax.plot(wl[mask], savgol_filter(np.abs(x3[mask]), 71, 1), alpha=1)  # /lamp[mask])

plt.savefig(savedir + "lockin_psd_tri.png", dpi=300)
plt.close()


phases = np.linspace(0,np.pi,360)
a = np.zeros((len(phases),ts.shape[0]))
for i,p in enumerate(phases):
    ref1 = np.sin(2 * np.pi * t2 * f + p + phase)
    x1 = lockin_filter(ts,ref1)
    x1 = savgol_filter(x1, 71, 1)
    a[i,:] = x1

plt.imshow(a, extent=(wl.min(), wl.max(), phases.max(), phases.min()), aspect=100,cmap=plt.get_cmap("seismic"))
plt.xlabel("Wavelength / nm")
plt.ylabel("Phase / rad")
#plt.axhline(y=phase)
plt.savefig(savedir + "lockin_phases", dpi=400)
plt.close()






# @jit()
# def func(x,a,f,p,c):
#     return a*np.sin(2 * np.pi * x * f + p) + c
#
# res = np.zeros(width)
# res_fit = np.zeros(width)
# res_phase = np.zeros(width)
# res_fit_phase = np.zeros(width)
# freqs = np.zeros(width)
# #for ind in np.arange(400,500):
# for ind in range(width):
#     print(ind)
#     fft = np.fft.rfft(ts[ind,:],norm="ortho")
#     d = np.absolute(fft)
#     p = np.angle(fft)
#     res[ind] = d[1:].max()
#     res_phase[ind] = p[d[1:].argmax()+1]
#
#     f_buf = np.fft.rfftfreq(t2.shape[0], d=t2[1] - t2[0])
#     freqs[ind] = f_buf[d[1:].argmax()+1]
#
# mask2 = wl > 0#(wl >= 400) & (wl <= 600)
# freqs = freqs[mask2]
# freqs = freqs[np.argwhere(res[mask2] > np.max(res[mask2])*0.9)]
# freqs = freqs[:,0]
# freqs = freqs[np.argwhere(freqs > 0)]
# print(freqs.shape)
#
# res_phase += np.pi
# phase = res_phase[np.argmax(res)]
#
# f = freqs[0]
# print("freq1: "+str(f))
#
# maxind = np.argmax(res)
# y = ts[maxind, :]
#
# fft = np.fft.rfft(y, norm="ortho")
# d = np.absolute(fft)
# p = np.angle(fft)
# f_buf = np.fft.rfftfreq(t2.shape[0], d=t2[1] - t2[0])
# print(f_buf[d[1:].argmax()])
#
# initial_guess = [y.max() - y.min(), f, 0, np.mean(y)]
# bounds = ([initial_guess[0] * 0.5, 0, 0, y.min()], [initial_guess[0] * 1.5, 1, 2 * np.pi, y.max()])
# #optimize_func = lambda x: np.sum(np.square(np.subtract(func(t2, x[0], x[1], x[2], x[3]), y)))
# #minimizer_kwargs = {"method": "SLSQP", "bounds": bounds}
# #popt = basinhopping(optimize_func, initial_guess, minimizer_kwargs=minimizer_kwargs)
# #popt = minimize(optimize_func, initial_guess, method="SLSQP")#, bounds=bounds)
# #popt = popt.x
# popt, pcov = curve_fit(func, t2, y, p0=initial_guess)#,bounds=bounds)
# #print(popt)
#
# fig = plt.figure()
# plt.plot(t2,y)
# plt.plot(t2,func(t2,popt[0],popt[1],popt[2],popt[3]))
# plt.show()
# #plt.savefig(savedir + "fit.png",dpi=300)
# #plt.close()
#
# f = popt[1]
# phase = popt[2]
# print("freq2: "+str(f))
#
# ref = np.sin(2 * np.pi * t2 * f + phase)
# ref2 = np.sin(2 * np.pi * t2 * f  + phase - np.pi/2)
#
# x2 = lockin_filter(ts,ref)
# y2 = lockin_filter(ts,ref2)
# res_amp = np.abs(x2 + 1j * y2)
# res_phase = np.angle(x2 + 1j * y2)
# maxind = np.argmax(savgol_filter(res_amp, 71, 1))
# phase = res_phase[maxind]
# print("phase1: "+str(phase))
#
# phases = np.linspace(0,2*np.pi,3600)
# x1 = np.zeros(len(phases))
# for i in range(len(phases)):
#     ref1 = np.sin(2 * np.pi * t2 * f + phases[i])
#     x1[i] = lockin_filter(np.matrix(ts[maxind,:]),ref1)
#
# plt.plot(phases,x1)
# plt.show()
#
#
# phase = phases[np.argmax(x1)]
# print("phase1: "+str(phase))
# phase = np.pi/2
#
# res_phase = np.angle( (x2*np.cos(phase)-y2*np.sin(phase)) + 1j * ( x2*np.sin(phase) + y2*np.cos(phase) ))
#
#
#
# phases = np.linspace(0,np.pi,360)
# a = np.zeros((len(phases),ts.shape[0]))
# for i,p in enumerate(phases):
#     ref1 = np.sin(2 * np.pi * t2 * f + p + phase)
#     x1 = lockin_filter(ts,ref1)
#     #x1 = savgol_filter(x1, 71, 1)
#     a[i,:] = x1
#
# plt.imshow(a, extent=(wl.min(), wl.max(), phases.max(), phases.min()), aspect=100,cmap=plt.get_cmap("seismic"))
# plt.xlabel("Wavelength / nm")
# plt.ylabel("Phase / rad")
# plt.axhline(y=np.pi/2)
# plt.show()
# #plt.savefig(savedir + "lockin_phases", dpi=400)
# #plt.close()
#
# ref = np.sin(2 * np.pi * t2 * f + phase - np.pi)
# ref2 = np.sin(2 * np.pi * t2 * f  + phase - np.pi/2)
# x2 = lockin_filter(ts,ref)
# x3 = lockin_filter(ts,ref2)
#
# print(wl[np.argmax(x2)])
# print(wl[np.argmax(x3)])
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# p1 = ax.plot(wl, x2)
# p2 = ax.plot(wl, x3)
# plt.show()
# #plt.savefig(savedir + "lockin_psd.png", dpi=300)
# #plt.close()
#
#
# s1 = spec1+spec2
# s2 = x2+x3
# plt.plot(wl,s1/s1.max())
# plt.plot(wl,s2/s2.max())
# plt.show()