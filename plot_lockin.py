import os
import re

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from scipy import interpolate

import matplotlib
#import seaborn as sns
matplotlib.use('QT4Agg')

#from plotsettings import *

import matplotlib.pyplot as plt
from matplotlib import cm
matplotlib.pyplot.switch_backend('QT4Agg')
#from mayavi import mlab


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

#path = '/home/sei/Spektren/p57m_did6_01/'
#path = '/home/sei/Spektren/p57m_did6_01_zlockin/'
#path = '/home/sei/Spektren/p57m_did6_01_map/'
#path = '/home/sei/Spektren/p57m_did6_01_zlockin/'
#path = '/home/sei/Spektren/p57m_did6_01_xlockin2/'
#path = '/home/sei/Spektren/p57m_did6_01_xlockin/'
#path = '/home/sei/Spektren/p57m_did6_01_xlockin_1um/'
#path = '/home/sei/Spektren/p57m_did6_01_ylockin2/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_0_7um/'
#path = '/home/sei/Spektren/p57m_did6_01_xlockin_1um/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_0_4um/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_0_3um/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_0_1um/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_through_0_5um/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_through_0_7um/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_through_0_9um/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_through_0_9um_2/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_through_0_9um_long/'
#path = '/home/sei/Spektren/p57m_did6_03_lockin_through_0_5um_long_long/'
#path = '/home/sei/Spektren/p52m_test/'
#path = '/home/sei/Spektren/p41m_dif0/'
#path = '/home/sei/Spektren/p41m_dif6/'
#path = '/home/sei/Spektren/p41m_dia0/'
#path = '/home/sei/Spektren/kegel4/'
#path = '/home/sei/Spektren/p57m_did6_01_zlockin/'
path = '/home/sei/Spektren/lockintest1/'


maxwl = 1000
minwl = 430

savedir = path


wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, counts = np.loadtxt(open(savedir + "normal.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

mask = (wl >= minwl) & (wl <= maxwl)
#wl = wl[mask]

lockin = np.loadtxt(open(savedir + "lockin.csv", "rb"), delimiter="\t")
lockin = lockin[:,1:]

#d = denoise_bilateral(lockin,multichannel=False)
#lockin = d


x = np.arange(0,lockin.shape[1])
y = np.arange(0,lockin.shape[0])
X, Y =  np.meshgrid(x, y)

plt.figure()
plt.pcolormesh(X,Y,lockin)
plt.savefig(savedir + "signal.png")
plt.close()

x2 = np.linspace(0,len(x),num=5000)
y2 = np.linspace(0,len(y),num=2000)
X2, Y2 = np.meshgrid(x2,y2)
spline = interpolate.interp2d(x, y, lockin,kind="cubic")#,kind="linear")
Z2 = spline(X2[0,:], Y2[:,0])
#
# plt.figure()
# plt.pcolormesh(X2,Y2,Z2)
# plt.savefig(savedir + "interpolated_signal.png")
# plt.close()

lockin = Z2

#for i in range(lockin.shape[1]):
#    lockin[:,i] /= (lamp-dark)


freq = 0.03

buf = np.linspace(0,len(x),len(x2))
#f = np.fft.rfftfreq(len(x))#,np.diff(x2)[0])
f = np.fft.rfftfreq(len(x2),np.diff(buf)[0])

#spline= interpolate.interp1d(np.arange(0,len(f)),f)
#f = spline(np.linspace(x2)

ind = (f > 0.0) & (f < 0.1)
#ind = (f > 0.019) & (f < 0.021)
#ind = 4

d1 = np.zeros(lockin.shape[0])
d2 = np.zeros(lockin.shape[0])
p2 = np.zeros(lockin.shape[0])
for i in range(lockin.shape[0]):
    l = lockin[i, :]
    a =np.fft.rfft(l)
    b = np.abs(a)
    p = np.angle(a)
    #max_ind = np.argmax(b[ind])
    #d1[i] = b[(f < freq+freq/100)][-1]
    ind = (f < freq*2+freq/100)
    d2[i] = b[ind][-1]
    p2[i] = p[ind][-1]

    #d2[i] = b[ind][max_ind]
    #p2[i] = p[ind][max_ind]


print(f[ind][-1])
plt.figure()
plt.plot(p2/p2.max())
plt.plot(d2/np.max(d2))
plt.savefig(savedir + "phase.png")
plt.close()


#max_ind = np.argmax


#d1 = d1[mask]
#d1 = d1/np.max(d1)

d4 = d2/(lamp-dark)
d4 = d4[mask]
d4 = d4/np.max(d4)

d2 = d2[mask]
d2 = d2/np.max(d2)

d3 = (counts-bg)/(lamp-dark)
d3 = d3[mask]
d3 = d3 - np.min(d3)
d3 = d3/np.max(d3)


#plt.plot(wl,d1)
plt.figure()
plt.plot(wl[mask],d2,label="lockin")
plt.plot(wl[mask],d4,label="lockin corr")
plt.plot(wl[mask],d3,label="mean spec")
plt.legend()
plt.savefig(savedir + "lockin_comparison.png")
plt.close()
#plt.show()


#num = 5000
#xl = np.arange(0, len(l))
#x2 = np.linspace(0, lockin.shape[1], num)

#f = np.fft.rfftfreq(len(x2),np.diff(x2)[0])
#f = np.fft.rfftfreq(lockin.shape[1],1/100)
Z = np.zeros((len(wl),len(f)))
P = np.zeros((len(wl),len(f)))
for i in range(len(wl)):
    l = lockin[i, :]
    #spline = interpolate.splrep(x=xl, y=l, s=0)
    #y2 = interpolate.splev(x2,spline,der=0)
    #a = np.fft.rfft(y2,norm='ortho')
    a = np.fft.rfft(l)#, norm='ortho')
    b = np.abs(a)
    p = np.angle(a)
    #f = np.fft.rfftfreq(len(l))
    #d1[i] = a[(f < freq+freq/100)][-1]
    #d2[i] = a[(f < freq*2+freq/100)][-1]
    Z[i,:] = b
    P[i,:] = p

ind = (f > 0.0) & (f < 0.1)
print(f[ind])


x = f[ind]
y = wl[mask]

Z = Z[:,ind]
Z = Z[mask,:]
Z = Z / np.max(Z)

P = P[:,ind]
P = P[mask,:]

X, Y =  np.meshgrid(x, y)

plt.figure()
plt.pcolormesh(X,Y,Z)
#plt.show()
plt.savefig(savedir + "fft_pcolor.png")
plt.close()

plt.figure()
plt.pcolormesh(X,Y,P)
#plt.show()
plt.savefig(savedir + "phase_pcolor.png")
plt.close()

#cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
#cmap = sns.cubehelix_palette( as_cmap=True, reverse=True)
colors = cm.viridis(np.linspace(0, 1, len(x)))
#colors = cmap(np.linspace(0, 1, len(x)))
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(wl[mask],d3,color="red",linewidth = 0.5)

for i in range(len(x)):
    plt.plot(y,Z[:,i],color=colors[i],linewidth=0.5)
    #ax.fill_between(y, Z[:,i], interpolate=True, color='blue')


plt.savefig(savedir + "fft_comparison.png")
plt.close()




# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X, Y =  np.meshgrid(x, y)
# #ax.plot_wireframe(X, Y, Z, rstride=5, cstride=10)
# #ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0,)
# ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0,)
# #N = P
# #N -= N.min()
# #N /= N.max()
# #ax.plot_surface(X, Y, Z,linewidth=0.1,facecolors=cm.jet(N),shade=False)#, cmap=cm.coolwarm)
# plt.show()
#
#
#
#
#
#X, Y =  np.meshgrid(np.arange(0,len(x)), np.arange(0,len(y)))
# N = P/P.max()
# color = cm.jet(N)
# mlab.surf(X.transpose(), Y.transpose(), Z, color = color, warp_scale="auto")#,representation='wireframe')
# mlab.surf(Z,warp_scale="auto")
# mlab.draw()
# mlab.show()
#
#
#X, Y =  np.meshgrid(x, y)
#xnew, ynew = np.meshgrid(np.linspace(0,len(x),num=2000), np.linspace(0,y.max(),num=2000))
#tck = interpolate.bisplrep(X, Y, Z, s=0)
#znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)




    #
#
#
#
#
#
#
#
#
#
# def ref_fun(x, f, p):
#     return np.cos(2 * np.pi * x * f + p)
#
#
# x = np.arange(0, lockin.shape[1])
# ref = ref_fun(x, 2*freq, np.pi)
# res1 = np.zeros(lockin.shape[0])
# for i in range(lockin.shape[0]):
#     #ref = np.cos(2 * np.pi * x * f+np.pi)
#     buf = ref * lockin[i, :]
#     buf = np.sum(buf)
#     res1[i] = buf
#
# plt.plot(res1/np.max(res1))
# plt.show()
# #plt.savefig(savedir + "lockin_standard.png")
# plt.close()
#
# ind = 780
# l = lockin[ind,:]
# l = l - np.min(l)
# l = l/np.max(l)
# plt.plot(l)
# plt.plot(ref/np.max(ref))
# plt.show()
#
#
# res2 = np.zeros(lockin.shape[0])
# for i in range(lockin.shape[0]):
#     x = np.arange(0, lockin.shape[1])
#     ref = np.cos(2 * np.pi * x * f)
#     buf = ref * lockin[i, :]
#     p0 = [np.max(buf)-np.min(buf),np.mean(buf)]
#     popt, pcov = curve_fit(cos_fit, x, buf, p0=p0)
#     res2[i] = -popt[0]
#
# plt.plot(res2)
# plt.savefig(savedir + "lockin_fit.png")
# plt.close()
#
#
# lockincorr = np.zeros(lockin.shape)
# for i in range(lockincorr.shape[1]):
#     lockincorr[:,i] = (lockin[:,i]-bg)/(lamp-dark)
#
# res3 = np.zeros(lockincorr.shape[0])
# for i in range(lockincorr.shape[0]):
#     x = np.arange(0, lockincorr.shape[1])
#     ref = np.cos(2 * np.pi * x * f)
#     buf = ref * lockincorr[i, :]
#     buf = np.sum(buf)
#     res3[i] = -buf
#
# plt.plot(res3)
# plt.savefig(savedir + "lockincorr_standard.png")
# plt.close()
#
#
# def cos_fit(x, amplitude, offset):
#     return (np.cos(2 * np.pi * x * f + 0)) ** 2 * amplitude + offset
#
# lockincorr[np.isneginf(lockincorr)] = 0
# lockincorr[np.isinf(lockincorr)] = 0
#
# res4 = np.zeros(lockincorr.shape[0])
# for i in range(lockincorr.shape[0]):
#     x = np.arange(0, lockincorr.shape[1])
#     ref = np.cos(2 * np.pi * x * f)
#     buf = ref * lockincorr[i, :]
#     p0 = [np.max(buf) - np.min(buf), np.mean(buf)]
#     popt, pcov = curve_fit(cos_fit, x, buf, p0=p0)
#     res4[i] = -popt[0]
#     #res4[i] = -np.sum(cos_fit(x,*popt))
#
#
# plt.plot(wl[mask], res4[mask])
# plt.savefig(savedir + "lockincorr_fit.png")
# plt.close()
#
#
#
#
#
# #
# # newfig(0.9)
# # plt.plot(wl[mask], counts[mask], linewidth=1)
# # plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.6)
# # plt.ylabel(r'$I_{df}\, /\, a.u.$')
# # plt.xlabel(r'$\lambda\, /\, nm$')
# # # plt.plot(wl, counts)
# # plt.tight_layout()
# # plt.savefig(savedir + "plots/" + files[i] + ".png", dpi=300)
# # plt.close()