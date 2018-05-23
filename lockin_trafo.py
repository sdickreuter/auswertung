import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter, sawtooth
#import harminv
from numba import jit
#from plotsettings import *
from scipy import signal
from scipy import ndimage
import math

import matplotlib.pyplot as plt
import timeit

@jit(nopython=True)
def lockin_filter(signal, freqs):
    width = signal.shape[0]
    x = np.arange(0,width,1)
    a = np.zeros((len(freqs),2))
    for i in range(len(freqs)):
        for j in range(width):
            a[i, 0] += signal[j] * math.sin(2*np.pi*freqs[i]*x[j])
            a[i, 1] += signal[j] * math.sin(2 * np.pi * freqs[i] * x[j]+np.pi/4)

        a[i,0] /= width
        a[i,1] /= width

    return a

@jit(nopython=True)
def lockin_filter2(signal, freqs):
    width = signal.shape[0]
    x = np.arange(0,width,1)
    a = np.zeros((len(freqs),2))

    for i in range(len(freqs)):
        a[i, 0] = np.sum( signal * np.sin(2*np.pi*freqs[i]*x))/width
        a[i, 1] = np.sum( signal * np.sin(2 * np.pi * freqs[i] * x+np.pi/4))/width

    return a


freqs = np.linspace(0.1,0.5,10)
phases = np.linspace(0,2*np.pi,10)

y = np.zeros(2000)
x = np.arange(0,len(y))
for i in range(len(freqs)):
    y += np.sin(2*np.pi*freqs[i]*x+phases[i])

f = np.fft.rfftfreq(x.shape[0])

l = np.zeros(len(f))
p = np.zeros(len(f))
res = lockin_filter(y,f)
a = res[:,0]
b = res[:,1]
l = np.abs(a + 1j * b)
p = np.angle(a + 1j * b)

def lockin():
    lockin_filter(y, f)

def lockin2():
    lockin_filter2(y, f)

def fft():
    np.fft.rfft(y, norm="ortho")

lockin()
lockin2()


print(timeit.timeit(lockin, number=10))
print(timeit.timeit(lockin2, number=10))
print(timeit.timeit(fft, number=10))



plt.figure()
plt.plot(f,l)
fft = np.absolute(np.fft.rfft(y, norm="ortho"))
plt.plot(f+0.01,fft/fft.max())
plt.show()

plt.figure()
plt.plot(f,p)
plt.plot(f,np.angle(np.fft.rfft(y, norm="ortho")))
plt.show()



# d = np.absolute(np.fft.rfft(lockin[ind, :]))
# p = np.abs(np.angle(np.fft.rfft(lockin[ind, :])))
# f = np.fft.rfftfreq(x.shape[0])