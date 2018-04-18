__author__ = 'sei'

import os
import re

import numpy as np

#from plotsettings import *

import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import matplotlib.widgets as mwidgets
import matplotlib as mpl

try:
    import cPickle as pickle
except ImportError:
    import pickle


path = '/home/sei/Spektren/pranoti/'


#samples = ['E10287 A1 5s highres']
#samples = ['E 10283 E9 0.0s map', 'E 10283 C1 0.2s map','E 10283 A1 1s map','E 10284 D7 10s map']
samples = ['E10287 A3 5s','E10287 A7 2s','E10287 A1 5s','E 10283 E9 0.0s map','E 10283 C1 0.2s map','E 10283 A1 1s map','E 10284 D7 10s map']
#samples = ['E10287 A7 2s']


wls = [400,500,600,700,800,900,1000]

def extents(f):
  delta = np.round(np.diff(np.sort(f)).max())
  #return [f.min() - delta/2, f.max() + delta/2]
  return [f.min(), f.max()]


letters = [chr(c) for c in range(65, 91)]

savedir = path + "wavelength_maps/"
try:
    os.mkdir(savedir )
except:
    pass

for sample in samples:
    print(sample)

    loaddir = path + sample + '/'

    wl, dark = np.loadtxt(open(loaddir + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

    sns.set_style("ticks", {'axes.linewidth': 1.0,
                            'xtick.direction': 'in',
                            'ytick.direction': 'in',
                            'xtick.major.size': 3,
                            'xtick.minor.size': 1.5,
                            'ytick.major.size': 3,
                            'ytick.minor.size': 1.5
                            })

    files = []
    for file in os.listdir(loaddir):
        if re.fullmatch(r"([A-Z]{1,5}[0-9]{1,5})(.csv)$", file) is not None:
            files.append(file)

    print(len(files))


    xy = np.zeros((len(files),2))
    for i in range(len(files)):
        file = files[i]
        meta = open(loaddir + file, "rb").readlines(300)
        xy[i, 0] = float(meta[11].decode())
        xy[i, 1] = float(meta[13].decode())
        #print('x: '+str(xy[i, 0])+'  y: '+str(xy[i, 1]))

    print(xy[:,0].min(),xy[:,0].max())
    print(xy[:, 1].min(), xy[:, 1].max())

    dx = np.round(np.diff(np.sort(xy[:,0])).max(),0)#,1)
    dy = np.round(np.diff(np.sort(xy[:,1])).max(),0)#,1)

    #dy = dx

    print((dx,dy))

    xy[:,0] -= xy[:,0].min()
    xy[:,0] /= dx
    xy[:,0] = np.round(xy[:,0])

    xy[:, 1] -= xy[:, 1].min()
    xy[:,1] /= dy
    xy[:, 1] = np.round(xy[:, 1])

    xy = np.array(xy,dtype=np.int)

    print(xy[:,0].min(),xy[:,0].max())
    print(xy[:, 1].min(), xy[:, 1].max())

    nx = xy[:,0].max()+1
    ny = xy[:, 1].max()+1



    img_all = np.zeros((nx, ny,len(wls)))
    index_matrix = np.zeros((nx, ny), dtype=np.int)
    for i in range(len(files)):
        index_matrix[xy[i, 0], xy[i, 1]] = i

    # mui importante!
    # use lower right corner as reference
    wl, lamp = np.loadtxt(open(loaddir + files[index_matrix[59, 0]], "rb"), delimiter=",", skiprows=16, unpack=True)
    print('Reference file: ' + files[index_matrix[59, 0]])

    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(loaddir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        for j in range(len(wls)):
            wl_ind = int(np.argmin(np.abs(wl - wls[j])))
            absorption = 1 - (counts[wl_ind] - dark[wl_ind]) / (lamp[wl_ind] - dark[wl_ind])
            img_all[xy[i,0],xy[i,1],j] = 1 - absorption

    cmap = plt.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=img_all.min(), vmax=img_all.max())
    norm = mpl.colors.Normalize(vmin=0, vmax=1.2)
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])

    for w,wavelength in enumerate(wls):
        print(wavelength)



        wl_ind = int(np.argmin(np.abs(wl-wavelength)))

        img = img_all[:,:,w]

        data_extent = (0, nx * dx, 0, ny * dy)

        print(extents(xy[:, 0]))
        print(extents(xy[:, 1]))

        plt.imshow(img.T, interpolation='nearest', cmap=cmap,norm=norm,
                   extent=extents(xy[:,0]) + extents(xy[:,1]), origin='lower')
        plt.xlim(extents(xy[:,0]))
        plt.ylim(extents(xy[:,1]))
        plt.xlabel(r'$x\, /\, \mu m$')
        plt.ylabel(r'$y\, /\, \mu m$')
        cb = plt.colorbar()
        cb.set_label(r'$T^{rel}_{'+str(np.round(wl[wl_ind]))+r'\,nm}$')
        plt.tight_layout()
        #plt.savefig(savedir + "overview/ "+str(np.round(wl[wl_ind]))+"nm_map.pdf", dpi=300)
        plt.savefig(savedir + sample + " " + str(int(np.round(wl[wl_ind])))+"nm.png", dpi=1200)
        plt.close()
