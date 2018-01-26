import pymesh
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter

#from plotsettings import *

import scipy.io as sio
import peakutils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
import matplotlib.cm as cm
#import seaborn as sns
from scipy.spatial import Delaunay
from scipy import signal
from scipy import interpolate
from matplotlib import gridspec


path = '/home/sei/MNPBEM/10degillu/'
sim = 'dimer_r45nm_d10nm_theta45.mat'


mat = sio.loadmat(path + sim)
p1 = mat['p1']
verts1 = p1['verts'][0][0]
faces1 = p1['faces'][0][0][:, 0:3]
faces1 = np.array(faces1 - 1, dtype=np.int)

mesh = pymesh.form_mesh(verts1, faces1)

pts = np.vstack(mesh.points) # (npoints, 2)-array of points
elements = np.vstack(mesh.elements) # (ntriangles, 6)-array specifying element connectivity

# Matplotlib's Triangulation module uses only linear elements, so use only first 3 columns when plotting
plt.triplot(pts[:,0], pts[:,1], elements[:,:3])
plt.show()