
import numpy as np
import matplotlib.pyplot as plt
import h5py


path= "/home/sei/Nextcloud/dipol/"
file = "dipol.mat"

filepath = path+file
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)

Ex = arrays["Ex"]
Ey = arrays["Ey"]
t = arrays["t"]
x = arrays["x"]
y = arrays["y"]

arrays = None

Ex = Ex[:,0,:,:]
Ey = Ey[:,0,:,:]
x = x[0,:]
y = y[0,:]
t = t[0,:]
#xx,yy = np.meshgrid(x,y)



#from scipy import interpolate
#ti = 400
#fEx = interpolate.interp2d(x[ti,:,:], y[ti,:,:], Ex[ti,:,:], kind='linear')
#fEy = interpolate.interp2d(x[ti,:,:], y[ti,:,:], Ey[ti,:,:], kind='linear')

from scipy.interpolate import RegularGridInterpolator

#f, ax = plt.subplots()

tmax = np.sqrt(7e-6**2 + 7e-6**2)/2.988e8
tmaxarg = np.argmin(np.abs(t-tmax))
tis = np.arange(100,tmaxarg,1)
peaks =  []
#peaks_rad =  []
#peaks_ =  []

for ti in tis:
    #ti = 100
    fEx = RegularGridInterpolator((y, x),Ex[ti,:,:])
    fEy = RegularGridInterpolator((y, x),Ey[ti,:,:])

    x2 = np.linspace(0, 7, 200) * 1e-6
    y2 = np.linspace(0, 7, 200) * 1e-6

    pts = np.array([x2, y2]).transpose()
    Ex2 = fEx(pts)
    Ey2 = fEy(pts)

    #Ex2 /= np.max(np.abs(Ex2))
    #Ey2 /= np.max(np.abs(Ey2))

    x2 *= 1e6
    y2 *= 1e6

    # plt.figure()
    # for i in range(Ex2.shape[0]):
    #     plt.arrow(x2[i],y2[i],Ex2[i],Ey2[i],width=1e1)
    #     plt.scatter(x2[i],y2[i])
    # plt.show()

    radial = np.zeros(len(Ex2))
    axial = np.zeros(len(Ex2))
    d = np.zeros(len(Ex2))
    for i in range(Ex2.shape[0]):
        radial[i] = Ex2[i]*np.sqrt(2) + Ey2[i]*np.sqrt(2)
        axial[i] = - Ex2[i] * np.sqrt(2) + Ey2[i] * np.sqrt(2)
        d[i] = np.sqrt(x2[i]**2 + y2[i]**2)

    #ax.plot(d,s)
    maxrad = np.argmax(np.abs(radial))
    peaks.append(np.abs(radial[maxrad]/axial[maxrad]))

peaks = np.array(peaks)
f, ax = plt.subplots()
ax.semilogy(t[tis]*2.998e8*1e6,peaks)
