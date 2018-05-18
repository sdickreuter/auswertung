import os
import re

import numpy as np

#from plotsettings import *

import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import Axes3D

path = '/home/sei/FDTD/Gitter/'

def extents(f):
  delta = np.round(np.diff(np.sort(f)).max())
  #return [f.min() - delta/2, f.max() + delta/2]
  return [f.min(), f.max()]

def load_mat(file):
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
    Pabs = Pabs*vol # absolute absorption
    return Pabs,x*1e3,y*1e3,z*1e3,freq

file = 'vert.mat'
Pabs,x,y,z,f = load_mat(path+file)
wl = 2.998e8/f

min_y = int(Pabs.shape[1]/2)-40
max_y = int(Pabs.shape[1]/2)+40

min_x = int(Pabs.shape[0]/2)-10
max_x = int(Pabs.shape[0]/2)+10
#min_x = 0
#max_x = 40


Pabs = Pabs[min_x:max_x,min_y:max_y,20:,2]
#y = y[min_y:max_y]

Pabs /= Pabs.max()
#thresh = 0.01
#Pabs[Pabs>thresh] = thresh
#Pabs /= Pabs.max()

print(z.max())
print(z.min())

print('z height :'+str(z.max()-z.min()))


# from pyevtk.hl import gridToVTK
# from pyevtk.vtk import VtkGroup
#
# gridToVTK(path+"./vtkdata", x, y, z, cellData = {"pabs" : Pabs}, pointData = {"pabs2" : Pabs})


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)


cube = Pabs
#def plot_cube(cube, angle=320):
cube = normalize(cube)

facecolors = plt.cm.viridis(cube)
facecolors[:, :, :, -1] = cube
facecolors = explode(facecolors)

filled = facecolors[:, :, :, -1] != 0
x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
ax = fig.gca(projection='3d')
#ax.view_init(30, angle)
ax.view_init(30, 320)
#ax.set_xlim(right=IMG_DIM * 2)
#ax.set_ylim(top=IMG_DIM * 2)
#ax.set_zlim(top=IMG_DIM * 2)

ax.voxels(x, y, z, filled, facecolors=facecolors)
plt.show()


#plot_cube(Pabs)


#
# font = {'family': 'sans',
#         'color':  'white',
#         'weight': 'normal',
#         'size': 18,
#         }
#
#
# fig = plt.figure(figsize=(8,3))
# grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
#                  nrows_ncols=(2,1),
#                  axes_pad=0.05,
#                  #share_all=True,
#                  #label_mode='L',
#                  cbar_location="right",
#                  cbar_mode="single",
#                  cbar_size="2%",
#                  cbar_pad=0.05,
#                  )
#
#
# #climit = [0,0.02]
#
# im0 = grid[0].imshow((Pabsh), interpolation='nearest', cmap=plt.get_cmap('jet'),
#                extent=extents(yh) + extents(xh))#, clim=climit)
# #grid[0].set_xlabel(r'$x\, /\, nm$')
# #grid[0].set_ylabel(r'$y\, /\, nm$')
# grid[0].text(0.03, 0.83, '(a)', horizontalalignment='center',verticalalignment='center', transform=grid[0].transAxes, fontdict=font)
# grid[0].set_axis_off()
#
# im1 = grid[1].imshow((Pabsv), interpolation='nearest', cmap=plt.get_cmap('jet'),
#                extent=extents(yv) + extents(xv))#, clim=climit)
# #grid[1].set_xlabel(r'$x\, /\, nm$')
# #grid[1].set_ylabel(r'$y\, /\, nm$')
# grid[1].text(0.03, 0.83, '(b)', horizontalalignment='center',verticalalignment='center', transform=grid[1].transAxes, fontdict=font)
# grid[1].set_axis_off()
#
# # cb = grid[0].cax.colorbar(im0)
# # grid[0].cax.set_xlabel(r'dissipated power [arb.u.]')#r'$T^{rel}_{400-700\,nm}$')
# # grid[0].cax.toggle_label(True)
# # #cb = plt.colorbar(ims[len(samples)-1],cax = grid[len(samples)-1].cax, orientation='horizontal')
# # grid[0].cax.set_xticks([0,0.5,1])
#
# grid[0].cax.colorbar(im0)
# cax = grid.cbar_axes[0]
# axis = cax.axis[cax.orientation]
# axis.label.set_text('dissipated power [arb.u.]')
# #cax.set_xticks([0,0.5,1])
#
# #plt.show()
# plt.savefig(path + "Fig5_2.png", dpi=1200)
# plt.savefig(path + "Fig5_2.eps", dpi=1200)
# plt.close()
#
# fig = plt.figure(figsize=(8,3))
# plt.title("(c)",loc="left")
# plt.semilogy(yh*1e3,absh*100)
# plt.semilogy(yv*1e3,absv*100)
# plt.ylabel(r'dissipated power $/\, \%$')
# plt.xlabel(r'$x /\, \mu m$')
# plt.legend(['p-polarisation','s-polarisation'])
# plt.tight_layout()
# plt.savefig(path + "line_comparison.png", dpi=1200)
# plt.savefig(path + "line_comparison.eps", dpi=1200)
# plt.close()
#
# print(np.sum(absh))
# print(np.sum(absv))