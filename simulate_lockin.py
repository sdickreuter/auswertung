import numpy as np

#from plotsettings import *
import matplotlib.pyplot as plt

import os
import re
import scipy.special as special
import scipy.signal as signal


path = '/home/sei/Spektren/p45m_did3/'
maxwl = 970
minwl = 420
wl, lamp = np.loadtxt(open(path + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, dark = np.loadtxt(open(path + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, bg = np.loadtxt(open(path + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

lamp = lamp-dark


def gauss(x, amplitude, xo, fwhm):
    sigma = fwhm / 2.3548
    xo = float(xo)
    g = amplitude * np.exp(-np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))
    return g.ravel()


def lorentz(x, amplitude, xo, sigma):
    xo = float(xo)
    g = amplitude * np.power(sigma / 2, 2.) / (np.power(sigma / 2, 2.) + np.power(x - xo, 2.))
    return g.ravel()

def airy(x, amplitude, xo):
    xo = float(xo)
    wl = 0.5
    NA = 0.3
    N = 1/(2*NA)
    g = amplitude*(special.j1(np.pi*x/(wl*N))/(np.pi*x/(wl*N)))**2
    return g.ravel()


x = np.linspace(-5,5,5000)

g = gauss(x,1,0,1)
l = lorentz(x,1,0,1)
a = airy(x,1,0)

c = np.zeros(len(a))
for i in range(len(a)):
    c += gauss(x,a[i],x[i],0.2)

c /= np.max(c)


#plt.plot(x,g)
#plt.plot(x,l)
#plt.plot(x,a)
#plt.plot(x,c)
#plt.show()

wl = np.linspace(400,900,2000)
spec = np.zeros(len(wl))
spec += lorentz(wl,1,550,80)
spec += lorentz(wl,4,690,200)
spec += 0.1


lamp = np.zeros(len(wl))
lamp += gauss(wl,60,670,200)
lamp += gauss(wl,10,850,100)
lamp += 1


#measurement = (lamp+np.random.normal(0,1,len(lamp)))*(spec+np.random.normal(0,0.4,len(spec)))
measured = lamp*spec

#plt.plot(wl,measured)
#plt.show()




f = 0.02
def ref(i,f):
    return np.cos(2 * np.pi * i * f)#+1

def ref2(i,f):
    return np.cos( (2 * np.pi * i) * f+np.pi)#+1


val = 100
d = np.zeros((val,len(measured)))
for i in range(val):
    pos = ref(i,f)*0.75
    #print(pos)
    #arg = np.min(np.argwhere(x > pos))
    #local_int = c[arg]
    local_int = airy(pos,1,0)
    d[i,:] = lamp*spec*local_int+np.random.normal(0,10,len(spec))

m = np.zeros((val,len(measured)))
for i in range(val):
    m[i,:] = lamp*spec*local_int+np.random.normal(0,10,len(spec))

plt.plot(d[:,280])
x = np.arange(0, d.shape[0])
plt.plot(ref2(x,f*2)*10)
plt.show()

def calc_lockin(d):
    res = np.zeros(d.shape[1])
    for i in range(d.shape[1]):
        x= np.arange(0,d.shape[0])
        buf = ref2(x,f*2) * d[:,i]
        #buf = np.cos(2 * np.pi * i * 2*f)* d[:,i]
        buf = np.sum(buf)
        res[i] = buf
    return res

l = calc_lockin(d)
m = np.mean(m,0)

mask = (wl >= 500) & (wl <= 800)


l = l/lamp
l = l/np.max(signal.savgol_filter(l[mask],51,1))

m = m/lamp
m = m/np.max(signal.savgol_filter(m[mask],51,1))
spec=spec/np.max(spec)
#plt.plot(wl,measured/np.max(measured),color="green")
plt.plot(wl,m-spec,color="gray")
plt.plot(wl,l-spec,color="blue")
#plt.plot(wl,spec,color="red")
plt.ylim((-1,1))
plt.show()