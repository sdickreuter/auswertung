__author__ = 'sei'

import os
import re

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter

from plotsettings import *

import matplotlib.pyplot as plt
from gauss_detect import *


#path = '/home/sei/Spektren/2C2/'
#sample = '2C2_75hept_B2'
#sample = '2C2_150hex_C2'
#sample = '2C2_150tri_A1'
#sample = '2C2_200hex_B1'
#sample = '2C2_200tri_A3'

path = '/home/sei/Spektren/2C1/'
#samples = ['2C1_200hex_A2']
#sample = '2C1_100hex_C2'
#sample = '2C1_150hept_B2'
#sample = '2C1_150hex_B3'
#sample = '2C1_200hex_A2'
samples = ['2C1_75hept_B2']


#path = '/home/sei/Spektren/'
#sample = 'SiOx2_mono_B_3'
#sample = 'SiOx2_mono_B_melt'
#sample = 'QDC24_4C'
#sample = 'QDC23_2D'

#path = '/home/sei/Spektren/rods/'
#sample = 'rods3_A0'
#sample = 'rods3_A0_melt'
#sample = 'rods3_A0_melt2'
#sample = 'rods3_A1'
#sample = 'rods3_A1_melt'
#sample = 'rods3_A2'
#sample = 'rods3_A2_melt'

#path = '/home/sei/Spektren/rods4/'
#samples = ['rods4_A0','rods4_A0m','rods4_A1','rods4_A1m','rods4_A2','rods4_A2m','rods4_B0','rods4_B0m','rods4_B1','rods4_B1m','rods4_B2','rods4_B2m']

#path = '/home/sei/Spektren/rods5/'
#samples = ['rods5_D0m','rods5_D0mm','rods5_D1m','rods5_D1mm']
#samples = ['rods5_D0m']
#samples = ['rods5_D0mmm','rods5_D1mmm']

#path = '/home/sei/Spektren/rods6/'
#samples = ['rods5_D0m','rods5_D0mm','rods5_D1m','rods5_D1mm']
#samples = ['rods5_D0m']
#samples = ['rods6_A1m','rods6_A1m_0pol','rods6_A1m_90pol']

#path = '/home/sei/Spektren/8/'
#samples = ['8_C3_horzpol','8_C4_horzpol','8_D2_horzpol','8_D3_horzpol','8_D3_vertpol']
#samples = ['8_C3_horzpol']



# grid dimensions
nx = 7
ny = 7
maxwl = 900
minwl = 450

for sample in samples:

    savedir = path + sample + '/'

    try:
        os.mkdir(savedir + "plots/")
    except:
        pass
    try:
        os.mkdir(savedir + "specs/")
    except:
        pass

    wl, lamp = np.loadtxt(open(savedir + "lamp.csv", "rb"), delimiter=",", skiprows=12, unpack=True)
    wl, dark = np.loadtxt(open(savedir + "dark.csv", "rb"), delimiter=",", skiprows=12, unpack=True)
    bg = dark
    #wl, bg = np.loadtxt(open(savedir + "background.csv", "rb"), delimiter=",", skiprows=12, unpack=True)

    mask = (wl >= minwl) & (wl <= maxwl)

    plt.plot(wl, lamp-dark)
    plt.savefig(savedir + "plots/lamp.png")
    plt.close()
    plt.plot(wl[mask], bg[mask])
    plt.savefig(savedir + "plots/bg.png")
    plt.close()
    plt.plot(wl[mask], dark[mask])
    plt.savefig(savedir + "plots/dark.png")
    plt.close()


    files = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([0-9]{5})(.csv)$", file) is not None:
            files.append(file)

    n = len(files)
    xy = np.zeros([n, 2])
    inds = np.zeros(n)
    files = np.array(files)

    for i in range(n):
        file = files[i]
        meta = open(savedir + file, "rb").readlines(300)
        xy[i, 0] = float(meta[7].decode())
        xy[i, 1] = float(meta[9].decode())
        inds[i] = i
        #wl, int[i] = np.loadtxt(open(savedir+file,"rb"),delimiter=",",skiprows=12,unpack=True)

    # reverse x values, because sample is upside down in microscope
    ordered = np.argsort(xy[:, 1])
    xy = xy[ordered, :]
    #xy[:, 0] = xy[::-1, 0]
    #xy[:, 1] = xy[::-1, 1]
    files = files[ordered]

    print(n)

    #plt.plot(xy[:,0],xy[:,1])
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
                #points[i + j * ny, :] = a0 + a * i + b * j
                #ids[i + j * ny] = (letters[nx - i - 1] + "{0:d}".format(j + 1))
                # points[i + j * ny, :] = a0 + a * i + b * j
                # ids[i + j * ny] = (letters[i] + "{0:d}".format(ny- j))
                points[i + j * ny, :] = a0 + a * i + b * j
                ids[i + j * ny] = (letters[i] + "{0:d}".format(j+1))
        ordered = np.argsort(points[:, 0])
        points = points[ordered, :]
        #points[:, :] = points[::-1, :]
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


    min_ind = np.argmin(xy[:, 1] * xy[:, 0])
    max_ind = np.argmax(xy[:, 1] * xy[:, 0])
    x0 = xy[min_ind, 0]
    y0 = xy[min_ind, 1]
    ax =0.7 * (xy[max_ind, 0] - x0) / (nx - 1)
    ay =0.7 * (xy[min_ind, 1] - y0) / (ny - 1)
    bx =0.7 * (xy[min_ind, 0] - x0) / (nx - 1)
    by =0.7 * (xy[max_ind, 1] - y0) / (ny - 1)

    # error function for minimizing
    def calc_error(params):
        grid, ids = make_grid(nx, ny, params[0], params[1], params[2], params[3], params[4], params[5])
        #grid, ids = make_grid(nx,ny,x0,y0,params[0],params[1],params[2],params[3])
        return grid_diff(xy, grid)


    start = np.array([x0, y0, ax, ay, bx, by])
    bnds = ((0, 2*x0), (0, 2*y0), (0, ax*nx), (-by*ny, by*ny), (-ax*nx, ax*nx), (0, by*ny))

    #res = minimize(calc_error, start, method='SLSQP',bounds=bnds,tol=1e-12, options={ 'disp': True, 'maxiter': 500})
    res = minimize(calc_error, start, method='L-BFGS-B', jac=False,bounds=bnds, options={'disp': True, 'maxiter': 500})
    #res = minimize(calc_error, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})

    grid, ids = make_grid(nx, ny, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5])

    plt.plot(grid[:,0],grid[:,1],"bx")
    plt.plot(xy[:,0],xy[:,1],"r.")
    for x, y, s in zip(grid[:,0],grid[:,1],ids):
        plt.text(x-1,y+1,s)
    #plt.show()
    plt.savefig(savedir + "grid2.pdf", format='pdf')
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(grid[:, 0], grid[:, 1], "bx")
    plt.plot(xy[:, 0], xy[:, 1], "r.")
    #plt.show()
    for x, y, s in zip(xy[:, 0], xy[:, 1], ids):
        plt.text(x - 1, y + 1, s)

    plt.savefig(savedir + "grid.pdf", format='pdf')
    plt.close()

    n = xy.shape[0]


    class Resonances():
        def __init__(self, id, amp, x0, sigma):
            self.id = id
            self.amp = amp
            self.x0 = x0
            self.sigma = sigma

    int532 = np.zeros(n)
    int581 = np.zeros(n)
    peak_wl = np.zeros(n)
    max_wl = np.zeros(n)
    resonances = np.array(np.repeat(None,n),dtype=object)
    searchmask = (wl >= 500) & (wl <= 700)

    #nxy = [[round(x,6),round(y,6)] for x,y in xy]
    for i in range(n):
        x = xy[i, 0]
        y = xy[i, 1]
        file = files[i]
        print(ids[i]+" "+file)
        wl, counts = np.loadtxt(open(savedir + file, "rb"), delimiter=",", skiprows=16, unpack=True)
        counts = (counts - bg) / (lamp - dark)

        counts[np.where(counts == np.inf)] = 0.0
        ind = np.argmax(counts)
        max_wl[i] = wl[ind]
        #counts = (counts - bg)
        #counts = counts - np.mean(counts[950:1023])
        #l = lamp - dark
        #l = l - np.mean(l[0:50])
        #counts = counts/l
        #counts = counts / lamp
        filtered = savgol_filter(counts, 41, 1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

        #plt.figure(figsize=(8, 6))
        newfig(0.9)
        plt.plot(wl[mask], counts[mask],linewidth=1)
        #plt.plot(wl[mask], filtered[mask],color="black",linewidth=0.6)
        plt.ylabel(r'$I_{df}\, /\, a.u.$')
        plt.xlabel(r'$\lambda\, /\, nm$')
        #plt.plot(wl, counts)
        plt.tight_layout()
        #plt.savefig(savedir + "plots/" + ids[i] + ".png",dpi=300)
        plt.savefig(savedir + "plots/" + ids[i] + ".eps",dpi=1200)

        plt.close()

    #
    #     #xx = wl[330:700]
    #     #yy = counts[330:700]
    #     #wl = wl[300:900]
    #     #counts = counts[300:900]
    #     xx = wl[searchmask]
    #     #yy = counts[searchmask]
    #     yy = filtered[searchmask]
    #     wl = wl[mask]
    #     counts = counts[mask]
    #     filtered = filtered[mask]
    #     #min_height = max(counts)*0.1#np.mean(counts[20:40])
    #     min_height = 0.001#0.006#0.015
    #     amp, x0, sigma = findGausses(xx,yy,min_height,30)
    #     plotGausses(savedir + "plots/" + ids[i] + "_gauss.png",wl, filtered, amp, x0, sigma)
    #     #amp, x0, sigma = fitLorentzes(xx, yy, amp, x0, sigma,min_height*5,2,200)
    #     #amp, x0, sigma = fitLorentzes_iter(xx,yy,min_height,2,200)
    #     #amp, x0, sigma = fitLorentzes2(xx, yy, min_height,5,180,max_iter=len(amp))
    #     #amp, x0, sigma = fitLorentzes3(xx, yy, amp, x0, sigma,min_height,2,200)
    #
    #     ##amp, x0, sigma = fitLorentzes_iter(xx, yy, min_height,5,200, 2)
    #
    #
    #     #amp, x0, sigma = findLorentzes(xx,yy,min_height,100)
    #     #print(len(amp))
    #     #plotLorentzes(savedir + "plots/" + ids[i] + "_lorentz_test.png",wl, counts, amp, x0, sigma)
    #     #amp, x0, sigma = fitLorentzes(xx, yy, amp, x0, sigma,min_height,20,500)
    #
    #     ##plotLorentzes(savedir + "plots/" + ids[i] + "_lorentz.png",wl, counts, amp, x0, sigma)
    #     #plotLorentzes(savedir + "plots/" + ids[i] + "_lorentz.png", wl, filtered, amp, x0, sigma)
    #
    #     #print(sigma)
    #     f = open(savedir + "specs/" + ids[i] + "_corr.csv", 'w')
    #     f.write("x" + "\r\n")
    #     f.write(str(x) + "\r\n")
    #     f.write("y" + "\r\n")
    #     f.write(str(y) + "\r\n")
    #     f.write("\r\n")
    #     f.write("wavelength,intensity" + "\r\n")
    #     for z in range(len(counts)):
    #         f.write(str(wl[z]) + "," + str(counts[z]) + "\r\n")
    #     f.close()
    #     ind = np.min(np.where(wl >= 531.5))
    #     int532[i] = counts[ind]
    #     ind = np.min(np.where(wl >= 580.5))
    #     int581[i] = counts[ind]
    #     resonances[i] = Resonances(ids[i],amp,x0,sigma)
    #     if len(x0) > 0:
    #         peak_wl[i] = x0[0]
    #     else:
    #         #peak_wl[i] = 0
    #         peak_wl[i] = wl[np.argmax(filtered)]
    #
    #
    # f = open(savedir+"peaks_532nm.csv", 'w')
    # f.write("x,y,id,max"+"\r\n")
    # for i in range(len(ids)):
    #     f.write(str(xy[i,0])+","+str(xy[i,1])+","+str(ids[i])+","+str(int532[i])+"\r\n")
    # f.close()
    #
    # f = open(savedir+"peaks_581nm.csv", 'w')
    # f.write("x,y,id,max"+"\r\n")
    # for i in range(len(ids)):
    #     f.write(str(xy[i,0])+","+str(xy[i,1])+","+str(ids[i])+","+str(int581[i])+"\r\n")
    # f.close()
    #
    # f = open(savedir+"peak_wl.csv", 'w')
    # f.write("x,y,id,peak_wl"+"\r\n")
    # for i in range(len(ids)):
    #     f.write(str(xy[i,0])+","+str(xy[i,1])+","+str(ids[i])+","+str(peak_wl[i])+"\r\n")
    # f.close()
    #
    # import pickle
    #
    # with open(savedir+r"resonances.pickle", "wb") as output_file:
    #     pickle.dump(resonances, output_file)
    #
    # newfig(0.9)
    # plt.hist(peak_wl,50)
    # plt.xlabel(r'$\lambda_{max}\, /\, nm$')
    # plt.ylabel('Häufigkeit')
    # # plt.plot(wl, counts)
    # plt.tight_layout()
    # plt.savefig(savedir + "hist.png", dpi=300)
    # plt.close()