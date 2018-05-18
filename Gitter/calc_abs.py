import os
import re

import numpy as np

#from plotsettings import *

import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
from mpl_toolkits.axes_grid1 import ImageGrid

path = '/home/sei/FDTD/Gitter/'

def load_mat1(file):
    f = sio.loadmat(file)

    Pabs = np.array(f['Pabs'])
    freq = np.array(f['f'])
    si = np.array(f['si'])
    sp = np.array(f['sp'])
    x = np.array(f['x'])[:,0]
    y = np.array(f['y'])[:,0]
    z = np.array(f['z'])[:,0]


    wl = 2.998e8/freq

    vol = np.diff(x)[0]*np.diff(y)[0]*np.diff(z)[0]

    #Pabs = Pabs.transpose(3,2,1,0)
    Pabs = np.sum(Pabs,axis=2)
    Pabs = Pabs*vol # absolute absorption
    return Pabs,x*1e3,y*1e3,f

def load_mat2(file):
    with h5py.File(file, 'r') as f:
        Pabs = np.array(f['Pabs'])
        freq = np.array(f['f'])
        si = np.array(f['si'])
        sp = np.array(f['sp'])
        x = np.array(f['x'])[0,:]
        y = np.array(f['y'])[0,:]
        z = np.array(f['z'])[0,:]


    wl = 2.998e8/freq

    vol = np.diff(x)[0]*np.diff(y)[0]*np.diff(z)[0]

    Pabs = Pabs.transpose(3,2,1,0)
    Pabs = np.sum(Pabs,axis=2)
    Pabs = Pabs*vol # absolute absorption
    return Pabs,x*1e3,y*1e3,f

listdir = os.listdir(path)
files = []
for file in listdir:
    if re.search(r"(.mat)$", file) is not None:
        files.append(file)

print(files)
Pabs = np.zeros((1,1,3))
for f in files:
    try:
        Pabs, x, y, fr = load_mat1(path + f)
    except Exception:
        try:
            Pabs, x, y, fr = load_mat2(path + f)
        except Exception:
            print("Error")

    if Pabs.shape[1] > 400:
        min_y = int(Pabs.shape[1] / 2) - 100
        max_y = int(Pabs.shape[1] / 2) + 100
        Pabs_red = Pabs[:, min_y:max_y, :]
        print(f + ' ' + str(round(np.sum(Pabs[:, :, 0]) * 100, 1)) + '%, klein: '+ str(round(np.sum(Pabs_red[:, :, 2]) * 100, 1)) + '%')
    else:
        print(f+' '+str(round(np.sum(Pabs[:,:,0])*100,1))+'%')

