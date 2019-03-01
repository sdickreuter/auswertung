import os
import sys

from matplotlib import ticker
from plotsettings import *
from scipy.signal import savgol_filter

from gridfit import fit_grid_spectra


def make_plotmap(path, nx, ny, minwl,maxwl,switch_xy=False,flip_x=False,flip_y=False):
    files, ids, xy = fit_grid_spectra(path, nx, ny,switch_xy,flip_x,flip_y)

    savedir = path
    wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=8, unpack=True)
    wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=8, unpack=True)
    # bg = dark
    wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=8, unpack=True)

    mask = (wl >= minwl) & (wl <= maxwl)

    n = xy.shape[0]

    max_counts = np.zeros(n)
    for i in range(n):
        file = files[i]
        # print(ids[i] + " " + file)
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=12, unpack=True)
        counts = (counts - bg) / (lamp - dark)
        max_counts[i] = counts[mask].max()



    dx = (np.max(xy[:, 0]) - np.min(xy[:, 0])) / (nx) * 0.8
    dy = (np.max(xy[:, 1]) - np.min(xy[:, 1])) / (ny) * 1.0

    # cm = plt.cm.get_cmap('RdYlBu')
    cm = plt.cm.get_cmap('rainbow')
    colors = cm(np.linspace(0, 1, len(wl[mask][::10])))

    # f = plt.figure()
    f = newfig(1.1)
    # ax = f.axes()
    for i in range(n):
        x = xy[i, 0]
        y = xy[i, 1]
        file = files[i]
        # print(ids[i] + " " + file)
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=12, unpack=True)
        counts = (counts - bg) / (lamp - dark)
        filtered = savgol_filter(counts, 41, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        nx = wl[mask]
        nx = nx - nx.min()
        nx = nx / nx.max()
        nx = nx * dx
        counts = counts[mask]
        counts = counts - counts.min()
        counts = counts / max_counts.max()
        counts = counts * dy
        plt.arrow(x, y, dx, 0, head_width=0.2, head_length=0.2)
        plt.arrow(x, y, 0, dy, head_width=0.2, head_length=0.2)
        plt.scatter(x + nx[::10], y + counts[::10], s=1, c=colors, edgecolors='none', cmap=cm)
        plt.text(x+dx*0.1,y+dy*0.8,ids[i],fontsize=6)

    m = plt.cm.ScalarMappable(cmap=cm)
    # m.set_array(np.linspace(0,1,len(wl)))
    m.set_array(wl[mask][::10])
    cb = plt.colorbar(m)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.tick_params(axis='y', direction='out')
    cb.set_label(r'$\lambda\, /\, nm$')


    plt.xlabel(r'$x\, /\, \mu m$')
    plt.ylabel(r'$y\, /\, \mu m$')
    plt.tight_layout(.5)
    # plt.show()
    plt.savefig(savedir + "specmap.png", dpi=300)
    plt.savefig(savedir + "specmap.pgf")
    plt.close()

    return files,ids,xy

if __name__ == "__main__":

    if len(sys.argv) == 3:
        path = sys.argv[1]
        sample = sys.argv[2]
    else:
        print("No paramters given, using default.")
        # RuntimeError("Too much/less arguments")
        path = '/home/sei/Spektren/2C1/'
        sample = '2C1_75hept_B2'

    # grid dimensions
    nx = 7
    ny = 7
    maxwl = 900
    minwl = 450

    # for sample in samples:

    savedir = path + sample + '/'

    try:
        os.mkdir(savedir + "plots/")
    except:
        pass
    try:
        os.mkdir(savedir + "specs/")
    except:
        pass

    make_plotmap(savedir, nx, ny, maxwl, minwl)
