__author__ = 'sei'

import os
import re

import numpy as np

#from plotsettings import *

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

try:
    import cPickle as pickle
except ImportError:
    import pickle

path = '/home/sei/Spektren/pranoti/'

#samples = ['E 10283 E9 0.0s map','E 10283 C1 0.2s map','E 10283 A1 1s map','E 10284 D7 10s map']
#x0s = [35,31,35,27]
#y0s = [34,31,32,32]

samples = ['E 10283 B9 0.0s map','E 10283 C1 0.2s map','E 10283 A1 1s map','E 10284 D7 10s map']
x0s = [35,31,35,27]
y0s = [34,31,32,32]


halfwidth = 25#16

seconds = []
for s in samples:
    search = re.search(r"([0-9]{0,2}[.]{0,1}[0-9]{1,2})(s)", s)
    if search is not None:
        seconds.append(search.group(0)[:-1]+' s')
    #elif re.search(r"(graphene)", file) is not None:

seconds[0] = 'graphene'

def extents(f):
  delta = np.round(np.diff(np.sort(f)).max())
  #return [f.min() - delta/2, f.max() + delta/2]
  return [f.min(), f.max()]

fig = plt.figure(figsize=(9.75, 3))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,4),
                 axes_pad=0.15,
                 share_all=True,
                 label_mode='L',
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )


ims = []

for i in range(len(samples)):
    sample = samples[i]
    with open(path + sample + '.pkl', 'rb') as fp:
        x, y, img = pickle.load(fp)

    has_line = False
    try:
        with open(path + sample[:-3] + 'roi_points.pkl', 'rb') as fp:
            points = pickle.load(fp)
        has_line = True
    except Exception as e:
        print(e)

    xmask = (x > (x0s[i] - halfwidth)) & (x < (x0s[i] + halfwidth))
    ymask = (y > (y0s[i] - halfwidth)) & (y < (y0s[i] + halfwidth))

    x_new = x[xmask]
    y_new = y[ymask]
    img = img[xmask,:]
    img = img[:,ymask]

    if has_line:
        x1 = points[0]-x_new.min()
        y1 = points[1]-y_new.min()
        x2 = points[2]-x_new.min()
        y2 = points[3]-y_new.min()


    #img -= img.ravel()[0]
    x_new = x_new - x_new.min()
    y_new = y_new - y_new.min()

    #data_extent = (x_new.min(), x_new.max(), y_new.min(), y_new.max())
    #ims.append( grid[i].imshow(img, extent=data_extent, cmap=plt.get_cmap('viridis'), interpolation="nearest", clim=[0,1],origin="lower") )
    ims.append( grid[i].imshow(img.T, interpolation='nearest', cmap=plt.get_cmap('cubehelix'),
               extent=extents(x_new) + extents(y_new), origin='lower', clim=[0,1.1]) )


    if has_line:
        #grid[i].plot([x[x1_ind], x[x2_ind]], [y[y1_ind], y[y2_ind]], 'k--')
        grid[i].plot([x1, x2], [y1, y2], 'k--',linewidth=0.8)

    grid[i].set_xlabel(r'$x\, /\, \mu m$')
    grid[i].set_ylabel(r'$y\, /\, \mu m$')
    grid[i].set_xlim([x_new.min(),x_new.max()])
    grid[i].set_ylim([y_new.min(), y_new.max()])
    grid[i].set_title(seconds[i])
    #if i == 0:
    #    grid[i].set_xlabel(r'$x\, /\, \mu m$')
    #    #grid[i].set_ylabel(r'$y\, /\, \mu m$')
    #else:
    #    grid[i].set_ylabel('')
    #    #grid[i].set_xlabel('')

grid.axes_llc
grid[3].cax.colorbar(ims[3])
grid[3].cax.set_ylabel(r'$T^{rel}_{400-700\,nm}$')
grid[3].cax.toggle_label(True)

#plt.tight_layout()
plt.savefig(path + "map_overview.png", dpi=1200)
plt.close()