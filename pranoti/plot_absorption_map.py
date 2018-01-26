__author__ = 'sei'

import os
import re

import numpy as np

#from plotsettings import *

import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import matplotlib.widgets as mwidgets

try:
    import cPickle as pickle
except ImportError:
    import pickle


path = '/home/sei/Spektren/pranoti/'

samples = ['E 10283 E9 0.0s map']
#samples = ['E 10283 B9 0.0s map', 'E 10283 C1 0.2s map','E 10283 A1 1s map','E 10284 D7 10s map']

fliplr = False
flipud = False
pickle_data = True
plot_pointspectra = False


minwl = 400
maxwl = 700

def extents(f):
  delta = np.round(np.diff(np.sort(f)).max())
  #return [f.min() - delta/2, f.max() + delta/2]
  return [f.min(), f.max()]

class GetIndices:

    def __init__(self,img):
        #figWH = (8,5) # in
        self.fig = plt.figure()#figsize=figWH)
        self.ax = self.fig.add_subplot(111)
        self.img = img
        self.ax.imshow(img.T,interpolation='nearest',
               origin='lower')
        self.xindices = []
        self.yindices = []

        self.cursor = mwidgets.Cursor(self.ax, useblit=True, color='k')
        #self.cursor.horizOn = False

        self.connect = self.ax.figure.canvas.mpl_connect
        self.disconnect = self.ax.figure.canvas.mpl_disconnect

        self.clickCid = self.connect("button_press_event",self.onClick)

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return int(idx)

    def onClick(self, event):
        if event.button == 1:
            if event.inaxes:
                indx = int(event.xdata)
                indy = int(event.ydata)
                self.xindices.append(indx)
                self.yindices.append(indy)
                self.ax.scatter(indx,indy,color="Red",marker="x",zorder=100,s=50)
                self.fig.canvas.draw()
        else:
            self.cleanup()

    def cleanup(self):
        self.disconnect(self.clickCid)
        plt.close()




letters = [chr(c) for c in range(65, 91)]

for sample in samples:
    print(sample)

    savedir = path + sample + '/'

    try:
        os.mkdir(savedir + "overview/")
    except:
        pass
    try:
        os.mkdir(savedir + "plots/")
    except:
        pass
    try:
        os.mkdir(savedir + "specs/")
    except:
        pass




    wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

    mask = (wl >= minwl) & (wl <= maxwl)

    sns.set_style("ticks", {'axes.linewidth': 1.0,
                            'xtick.direction': 'in',
                            'ytick.direction': 'in',
                            'xtick.major.size': 3,
                            'xtick.minor.size': 1.5,
                            'ytick.major.size': 3,
                            'ytick.minor.size': 1.5
                            })

    files = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([A-Z]{1,2}[0-9]{1,2})(.csv)$", file) is not None:
            files.append(file)

    print(len(files))


    # #wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
    # # first measurement as reference !
    # wl, lamp = np.loadtxt(open(savedir + files[0], "rb"), delimiter=",", skiprows=16, unpack=True)
    #
    # plt.plot(wl, lamp-dark)
    # plt.xlim((minwl, maxwl))
    # plt.savefig(savedir + "overview/lamp.pdf", dpi=300)
    # plt.close()
    #
    # plt.plot(wl[mask], dark[mask])
    # plt.xlim((minwl, maxwl))
    # plt.savefig(savedir + "overview/dark.pdf", dpi=300)
    # plt.close()

    # for i in range(len(files)):
    #     file = files[i]
    #     wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
    #     counts = 1 - (counts - dark) / (lamp - dark)
    #
    #     counts[np.where(counts == np.inf)] = 0.0
    #     filtered = savgol_filter(counts, 27, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    #     newfig(0.9)
    #     plt.plot(wl[mask], counts[mask], color="grey", linewidth=0.5)
    #     plt.plot(wl[mask], filtered[mask], color="black", linewidth=0.5)
    #     plt.ylabel(r'$apsorption$')
    #     plt.xlabel(r'$\lambda\, /\, nm$')
    #     plt.xlim((minwl, maxwl))
    #     #plt.ylim([np.min(filtered[mask]), np.max(filtered[mask]) * 1.2])
    #     plt.ylim([0, 1])
    #     plt.tight_layout()
    #     plt.savefig(savedir + "plots/" + file[:-4] + ".pdf", dpi=300)
    #     #plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
    #     plt.close()
    #
    #     f = open(savedir + "specs/" + file[:-4] + ".csv", 'w')
    #     f.write("wavelength,intensity" + "\r\n")
    #     for z in range(len(counts)):
    #         f.write(str(wl[z]) + "," + str(counts[z]) + "\r\n")
    #     f.close()

    xy = np.zeros((len(files),2))
    for i in range(len(files)):
        file = files[i]
        meta = open(savedir + file, "rb").readlines(300)
        xy[i, 0] = float(meta[11].decode())
        xy[i, 1] = float(meta[13].decode())

    dx = np.round(np.diff(np.sort(xy[:,0])).max())
    dy = np.round(np.diff(np.sort(xy[:,1])).max())

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

    img = np.zeros((nx,ny))
    index_matrix = np.zeros((nx,ny),dtype=np.int)
    for i in range(len(files)):
        index_matrix[xy[i,0],xy[i,1]] = i

    # use lower right corner as reference
    wl, lamp = np.loadtxt(open(savedir + files[index_matrix[59,0]], "rb"), delimiter=",", skiprows=16, unpack=True)
    print('Reference file: '+files[index_matrix[59,0]])


    for i in range(len(files)):
        file = files[i]
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        absorption = 1 - np.sum((counts[mask] - dark[mask])) / np.sum((lamp[mask] - dark[mask]))
        img[xy[i,0],xy[i,1]] = 1 - absorption



    if fliplr:
        img = np.fliplr(img)
    if flipud:
        img = np.flipud(img)

    data_extent = (0, nx * dx, 0, ny * dy)

    print(extents(xy[:, 0]))
    print(extents(xy[:, 1]))

    plt.imshow(img.T, interpolation='nearest', cmap=plt.get_cmap('viridis'),
               extent=extents(xy[:,0]) + extents(xy[:,1]), origin='lower')
    plt.xlim(extents(xy[:,0]))
    plt.ylim(extents(xy[:,1]))
    plt.xlabel(r'$x\, /\, \mu m$')
    plt.ylabel(r'$y\, /\, \mu m$')
    cb = plt.colorbar()
    cb.set_label(r'$T^{rel}_{'+str(minwl)+'-'+str(maxwl)+r'\,nm}$')
    plt.tight_layout()
    plt.savefig(savedir + "overview/" + "map.pdf", dpi=300)
    plt.savefig(path + sample + ".png", dpi=1200)
    plt.close()

    x = np.linspace(0, img.shape[0],img.shape[0]) + dx
    y = np.linspace(0,img.shape[1],img.shape[1]) + dy

    print(img.shape)
    print(x.shape)
    print(y.shape)

    if pickle_data:
        with open(path + sample + '.pkl', 'wb') as fp:
            pickle.dump((x,y,img), fp)


    if plot_pointspectra:


        xyIndices = GetIndices(img)
        plt.show()
        plt.close()
        xindices = np.array(xyIndices.xindices,dtype=np.int)
        yindices = len(y)-np.array(xyIndices.yindices,dtype=np.int)


        x0 = xindices
        y0 = yindices

        #colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(x0)))
        colors = plt.cm.get_cmap('tab10').colors[:len(x0)]

        f, (ax1, ax2) = plt.subplots(2, 1)

        im = ax1.imshow(img.T, interpolation='nearest', cmap=plt.get_cmap('viridis'),
               extent=extents(xy[:,0]) + extents(xy[:,1]), origin='lower')
        ax1.set_xlim(0, nx * dx)
        ax1.set_ylim(0, ny * dy)
        cb = f.colorbar(im, ax=ax1)
        cb.set_label(r'$T^{rel}_{'+str(minwl)+'-'+str(maxwl)+r'\,nm}$')
        for i in range(len(x0)):
            #plt.plot(x0[i],y0[i],'o')
            ax1.scatter(x0[i]+1,y0[i]-1, s=30, facecolors='none',edgecolors=colors[i],linewidths=1.5)
        ax1.set_xlabel(r'$x\, /\, \mu m$')
        ax1.set_ylabel(r'$y\, /\, \mu m$')

        for i in range(len(x0)):
            xind = int(np.round(x0[i] / dx))
            yind = ny-int(np.round(y0[i] / dy))
            print(index_matrix[xind,yind])
            img[xind,yind] = 0
            file = files[index_matrix[xind,yind]]
            wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
            counts = 1 - (counts - dark) / (lamp - dark)
            ax2.plot(wl,1-counts,color=colors[i],zorder=10-i)

        ax2.set_ylabel(r'$T^{rel}_{'+str(minwl)+'-'+str(maxwl)+r'\,nm}$')
        ax2.set_xlabel(r'$\lambda\, /\, nm$')
        ax2.set_xlim((minwl, maxwl))
        #ax2.set_ylim([0, 1])
        #plt.legend(['inside electrode','edge of electrode','contact lead'])
        plt.savefig(savedir + "overview/" + sample + " point spectra.pdf", dpi=300)
        plt.close()

        plt.imshow(img.T, interpolation='nearest', cmap=plt.get_cmap('viridis'),
                   extent=extents(xy[:, 0]) + extents(xy[:, 1]), origin='lower')
        plt.xlim(0, nx * dx)
        plt.ylim(0, ny * dy)
        plt.tight_layout()
        plt.savefig(savedir + "overview/" + "map_check.pdf", dpi=300)
        plt.close()


    #
    # plt.imshow(img.T,extent=data_extent,cmap=plt.get_cmap('viridis'),interpolation="nearest")
    # cb = plt.colorbar()
    # cb.set_label(r'$T^{rel}_{'+str(minwl)+'-'+str(maxwl)+r'\,nm}$')
    # for i in range(len(x0)):
    #     #plt.plot(x0[i],y0[i],'o')
    #     plt.scatter(x0[i]+1,y0[i]-1, s=30, facecolors='none',edgecolors=colors[i],linewidths=1.5)
    # plt.xlabel(r'$x\, /\, \mu m$')
    # plt.ylabel(r'$y\, /\, \mu m$')
    # plt.tight_layout()
    # plt.savefig(savedir + "overview/" + "map_withpoints.pdf", dpi=300)
    # plt.close()
    #
    # for i in range(len(x0)):
    #     xind = int(np.round(x0[i] / dx))
    #     yind = ny-int(np.round(y0[i] / dy))
    #     print(index_matrix[xind,yind])
    #     img[xind,yind] = 0
    #     file = files[index_matrix[xind,yind]]
    #     wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
    #     counts = 1 - (counts - dark) / (lamp - dark)
    #     plt.plot(wl,1-counts,color=colors[i],zorder=10-i)
    #
    # plt.ylabel(r'$T^{rel}_{'+str(minwl)+'-'+str(maxwl)+r'\,nm}$')
    # plt.xlabel(r'$\lambda\, /\, nm$')
    # plt.xlim((minwl, maxwl))
    # plt.ylim([0, 1])
    # plt.tight_layout()
    # #plt.legend(['inside electrode','edge of electrode','contact lead'])
    # plt.savefig(savedir + "overview/" + "transmittance_spectra.pdf", dpi=300)
    # plt.close()
