__author__ = 'sei'

import os

import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import scipy.optimize as opt
#from scipy.special import *
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
import pandas as pd
import pickle
from matplotlib.mlab import griddata

#path = '/home/sei/Raman/2C2/'
#sample = '2C2_75hept_B2'
#sample = '2C2_150hex_C2'
#sample = '2C2_150tri_A1'
#sample = '2C2_200hex_B1'
#sample = '2C2_200tri_A3'

path = '/home/sei/Raman/2C1/'
sample = '2C1_75hept_B2'
#sample = '2C1_100hex_C2'
#sample = '2C1_150hept_B2'
#sample = '2C1_150hex_B3'
#sample = '2C1_200hex_A2'


savedir = path + sample + '_plots/'

#grid dimensions
nx = 7
ny = 7

try:
    os.mkdir(savedir)
except:
    pass

#data = np.loadtxt(path + sample + "_map.csv", delimiter=" ")
#data = np.transpose(data)
#print(data.shape)


with open(path + sample+'_positions.pkl', 'rb') as fp:
    x, y, files = pickle.load(fp)

y = np.flipud(y)
x-=x.min()
y-=y.min()

min_x = 0
max_x = 24
min_y = 0
max_y = 25

new_x = []
new_y = []
new_files = []
for i in range(len(x)):
    if x[i] >=min_x:
        if x[i] <= max_x:
            if y[i] >=min_y:
                if y[i] <= max_y:
                    new_x.append(x[i])
                    new_y.append(y[i])
                    new_files.append(files[i])

x = np.array(new_x)
y = np.array(new_y)
files = np.array(new_files)

x-=x.min()
y-=y.min()


inten = np.zeros(len(files))
for i in range(len(files)):
    wl, counts = np.loadtxt(open(path+sample+'/'+files[i], "rb"), delimiter="\t", skiprows=1, unpack=True)
    inten[i] = counts[(wl > 1580) & (wl < 1590)].max()

# xi = np.linspace(x.min(), x.max(), 500)
# yi = np.linspace(y.min(), y.max(), 500)
# # grid the data.
# zi = griddata(x, y, inten, xi, yi, interp='linear')
# # contour the gridded data, plotting dots at the nonuniform data points.
# #CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
# CS = plt.contourf(xi, yi, zi, 100,
#                   vmax=inten.max(), vmin=inten.min())
# plt.show()

with open(path + sample + '_positions_redo.pkl', 'wb') as fp:
    pickle.dump((x, y, files), fp, protocol=2)


xu = np.unique(x)
yu = np.unique(y)

data = np.zeros((len(xu),len(yu)))
for i in range(len(xu)):
    for j in range(len(yu)):
        data[i,j] = inten[(j-1)*len(xu)+i]

data = np.fliplr(data)
data = data.T

data_extent=( 0,data.shape[1]*0.2, 0, data.shape[0]*0.2)


thresh = threshold_otsu(data)

fdata = filters.gaussian_filter(data, sigma=1.5)

labeled, n = ndimage.label(fdata > thresh / 5)
xy = np.array(ndimage.center_of_mass(fdata, labeled, range(1, n + 1)))

xy = xy[:, [1, 0]]
xy = xy*0.2

#plt.imshow(labeled,extent=data_extent,origin="lower",interpolation="nearest")
plt.imshow(data,extent=data_extent,origin="lower",interpolation="nearest")
#plt.plot(xy[:,0],xy[:,1],"r.")
#plt.show()

# calculate grid points
def make_grid(nx, ny, x0, y0, ax, ay, bx, by):
    letters = [chr(c) for c in range(65, 91)]
    a0 = np.array([x0, y0])
    a = np.array([ax, ay])
    b = np.array([bx, by])
    points = np.zeros([nx * ny, 2])
    ids = np.empty(nx * ny, dtype=object)
    for i in range(nx):
        for j in range(ny):
            #print(j*ny+i)
            points[i + j * ny, :] = a0 + a * i + b * j
            ids[i + j * ny] = (letters[nx - i - 1] + "{0:d}".format(j+1))
    ordered = np.argsort(points[:, 0])
    points = points[ordered, :]
    points[:, :] = points[::-1, :]
    ids = ids[ordered]
    return points, ids


#calculate min dist between sets of points
def calc_mindists(points1, points2):
    dists = np.zeros(points1.shape[0])
    indices = np.zeros(points1.shape[0], dtype=np.int)
    buf = np.zeros(points2.shape[0])
    weights = np.zeros(points2.shape[0])
    for i in range(points1.shape[0]):
        for j in range(points2.shape[0]):
            #buf[j] = np.sqrt( np.sum( np.square( points2[j,:] - points1[i,:] ) ) )
            buf[j] = np.sum(np.square(points2[j, :] - points1[i, :]))
        indices[i] = np.argmin(buf)
        weights[indices[i]] += 1;
        dists[i] = buf[indices[i]] * weights[indices[i]]
    return dists, indices


# function for adding up distances of points between two grids
def grid_diff(points1, points2):
    return np.sum(calc_mindists(points1, points2))


# error function for minimizing
def calc_error(params):
    grid, ids = make_grid(nx, ny, params[0], params[1], params[2], params[3], params[4], params[5])
    return grid_diff(xy, grid)


cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=False)


min_ind = np.argmin(xy[:, 1] * xy[:, 0])
max_ind = np.argmax(xy[:, 1] * xy[:, 0])
x0 = xy[min_ind, 0]
y0 = xy[min_ind, 1]
ax = 1 * (xy[max_ind, 0] - x0) / (nx - 1)
ay = 1 * (xy[min_ind, 1] - y0) / (ny - 1)
bx = 1 * (xy[min_ind, 0] - x0) / (nx - 1)
by = 1 * (xy[max_ind, 1] - y0) / (ny - 1)

start = np.array([x0, y0, ax, ay, bx, by])
#res = opt.minimize(calc_error, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
res = opt.minimize(calc_error, start, method='L-BFGS-B', jac=False, options={'disp': True, 'maxiter': 500})

#res = fmin_bfgs(calc_error, start)
#grid, ids = make_grid(nx,ny,start[0],start[1],start[2],start[3],start[4],start[5])

grid, ids = make_grid(nx, ny, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5])
#grid, ids = make_grid(nx,ny,res[0],res[1],res[2],res[3],res[4],res[5])
d, inds = calc_mindists(grid, xy)

xy = xy[inds, :]
validpoints = np.where(d < 5)[0]

ids = ids[validpoints]
grid = grid[validpoints, :]
xy = xy[validpoints, :]

plt.plot(grid[:, 0], grid[:, 1], "bx")
plt.plot(xy[:, 0], xy[:, 1], "r.")
plt.show()

with open(path + sample + '_grid.pkl', 'wb') as fp:
    pickle.dump((xy[:,0], xy[:,1], ids), fp)