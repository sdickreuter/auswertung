__author__ = 'sei'

import os

import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import scipy.optimize as opt
from scipy.special import *
from plotsettings import *
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.filters import threshold_otsu


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


def plot_particles(image, fname):
    # Generate the markers as local maxima of the distance to the background
    distance = ndimage.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=image)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
    ax0, ax1, ax2 = axes
    ax0.imshow(image, interpolation='nearest')
    ax0.set_title('Overlapping objects')
    ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
    ax1.set_title('Distances')
    ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
    ax2.set_title('Separated objects')
    for ax in axes:
        ax.axis('off')
    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                        right=1)
    #plt.show()
    plt.savefig(fname, format='png')
    plt.close()


try:
    os.mkdir(savedir)
except:
    pass

data = np.loadtxt(path + sample + "_map.csv", delimiter=" ")
data = np.transpose(data)
print(data.shape)


#data = data[:, :140]
data = data[20:140, :130]
#data = data[10:290, :250]
#data = data[:, :130]
data = np.flipud(data)

data_extent=( 0,data.shape[1]*0.2, 0, data.shape[0]*0.2)


thresh = threshold_otsu(data)

fdata = filters.gaussian_filter(data, sigma=1.5)

labeled, n = ndimage.label(fdata > thresh / 4)
xy = np.array(ndimage.center_of_mass(fdata, labeled, range(1, n + 1)))

xy = xy[:, [1, 0]]
xy = xy*0.2


plt.imshow(labeled,extent=data_extent,origin="lower",interpolation="nearest")
plt.plot(xy[:,0],xy[:,1],"r.")
plt.show()

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


fig, ax = plt.subplots()
cax = ax.imshow(data,cmap=cmap,extent=data_extent,origin="lower", interpolation='nearest')
cb = plt.colorbar(cax)
cb.set_label(r'$I_{\nu}\, / \,a.u.$')
for x, y, s in zip(grid[:, 0], grid[:, 1], ids):
    ax.text(x - 0.5, y + 1.0, s, fontsize=8)

ax.set_xlabel(r'$x\, /\, \mu m$')
ax.set_ylabel(r'$y\, /\, \mu m$')
plt.tight_layout()
plt.savefig(savedir + "grid.eps", dpi=1200)
plt.savefig(savedir + "grid.pgf")
plt.close()


fig, ax = plt.subplots()
cax = ax.imshow(data,cmap=cmap,extent=data_extent,origin="lower", interpolation='nearest')
cb = plt.colorbar(cax)
cb.set_label(r'$I_{\nu}\, / \,a.u.$')
ax.set_xlabel(r'$x\, /\, \mu m$')
ax.set_ylabel(r'$y\, /\, \mu m$')
plt.tight_layout()
plt.savefig(savedir + "grid_pure.eps", dpi=1200)
plt.savefig(savedir + "grid_pure.pgf")
plt.close()



# from http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def twoD_Lorentz(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*a/(a**2 + (((x-xo)**2) + ((y-yo)**2)))
    return g.ravel()


def twoD_Gaussiansqr(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * (np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))**2)
    return g.ravel()


def twoD_Airy(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * (np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))**2)
    return g.ravel()


try:
    os.mkdir(savedir + "sections/")
except:
    pass

try:
    os.mkdir(savedir + "gaussfit/")
except:
    pass


n = xy.shape[0]
ram = np.zeros(n)
ramg = np.zeros(n)
stddev = np.zeros(n)
width = 2.0
for i in range(n):
    print(ids[i])
    x = xy[i, 0]
    y = xy[i, 1]
    # print((x,y))
    if x - width > 0:
        xstart = (x - width)/0.2
    else:
        xstart = 0
    if x + width < data.shape[1]*0.2:
        xstop = (x + width)/0.2
    else:
        xstop = data.shape[1]-1
    if y - width > 0:
        ystart = (y - width)/0.2
    else:
        ystart = 0
    if y + width < data.shape[0]*0.2:
        ystop = (y + width)/0.2
    else:
        ystop = data.shape[0]-1

    xstart = np.int(round(xstart))
    xstop = np.int(round(xstop))
    ystart = np.int(round(ystart))
    ystop = np.int(round(ystop))
    # print((xstart,xstop,ystart,ystop))
    # print((xstart*0.2,xstop*0.2,ystart*0.2,ystop*0.2))
    # print(data.shape)
    # print((ystart,ystop,data.shape[1]-xstop,data.shape[1]-xstart))

    #sub = data[ystart:ystop, data.shape[1]-xstop:data.shape[1]-xstart]
    #sub = data[xstart:xstop, ystart:ystop]
    sub = data[ystart:ystop, xstart:xstop]

    # print(sub.shape)

    fig, ax = plt.subplots()
    cax = ax.imshow(sub, cmap=cmap,extent=(xstart*0.2,xstop*0.2,ystart*0.2,ystop*0.2),origin="lower", interpolation='nearest')
    cb = plt.colorbar(cax)
    cb.set_label(r'$I_{\nu}\, / \,a.u.$')
    ax.plot(x,y)
    ax.set_xlabel(r'$x\, /\, \mu m$')
    ax.set_ylabel(r'$y\, /\, \mu m$')
    plt.tight_layout()

    #plt.axis('off')
    #plt.savefig(savedir + "sections/" + ids[i] + ".png")
    plt.tight_layout()
    plt.savefig(savedir + "sections/" + ids[i] + ".eps",dpi=1200)
    plt.savefig(savedir + "sections/" + ids[i] + ".pgf")
    plt.close()

    ram[i] = np.max(sub)
    nx = xstop-xstart
    ny = ystop-ystart
    xvalues = np.linspace(xstart*0.2,xstop*0.2,nx)
    yvalues =  np.linspace(ystart*0.2,ystop*0.2,ny)
    xx,yy = np.meshgrid(xvalues, yvalues)
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    ydata = sub.ravel()
    #   #def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    #initial_guess = (np.max(xdata[1,:]), x-xstart*0.2, y-ystart*0.2 ,3, 3,0, np.min(sub))
    initial_guess = (np.max(xdata[1,:]), x, y ,1, 1,0, np.min(sub))
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, ydata, p0=initial_guess)

    data_fitted = twoD_Gaussian((xx, yy), *popt)

    xvalues2 = np.linspace(xstart*0.2,xstop*0.2,nx*100)
    yvalues2 =  np.linspace(ystart*0.2,ystop*0.2,ny*100)
    xx2,yy2 = np.meshgrid(xvalues2, yvalues2)
    data_fitted2 = twoD_Gaussian((xx2, yy2), *popt)

    #   #twoD_Lorentz(xdata_tuple, amplitude, xo, yo, s_x, s_y, theta, offset):
    #initial_guess = (np.max(sub),y-ystart, x-xstart, 1, 1, 0, np.min(sub))
    #popt, pcov = opt.curve_fit(twoD_Lorentz, xdata, ydata, p0=initial_guess)
    #data_fitted = twoD_Lorentz((xx, yy), *popt)
    #   #def twoD_Gaussiansqr(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    ##initial_guess = (y-ystart, x-xstart, np.mean(xdata[1,:]),3, 3,0, np.min(sub))
    ##popt, pcov = opt.curve_fit(twoD_Gaussiansqr, xdata, ydata, p0=initial_guess)
    ##data_fitted = twoD_Gaussiansqr((xx, yy), *popt)
    #plot sub with gauss
    fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(sub, origin='bottom',
        extent=(xstart * 0.2, xstop * 0.2, ystart * 0.2, ystop * 0.2), interpolation='nearest',cmap=cmap)
    ax.contour(xx2, yy2, data_fitted2.reshape(xx2.shape), 4, colors='black',linewidths=0.5,linestyles="dashed")
    cb = plt.colorbar(cax)
    cb.set_label(r'$I_{\nu}\, / \,a.u.$')
    plt.xlabel(r'$x\, /\, \mu m$')
    plt.ylabel(r'$y\, /\, \mu m$')
    plt.tight_layout()
    #plt.savefig(savedir + "gaussfit/" + ids[i] + ".png", format='png')
    plt.savefig(savedir + "gaussfit/" + ids[i] + "_gauss.eps", dpi=1200)
    plt.savefig(savedir + "gaussfit/" + ids[i] + "_gauss.pgf")
    plt.close()

    fig, ax = plt.subplots()
    cax = ax.imshow(sub-data_fitted.reshape(xx.shape), origin='bottom',
        extent=(xstart * 0.2, xstop * 0.2, ystart * 0.2, ystop * 0.2),interpolation='nearest',cmap=cmap)
    ax.contour(xx2, yy2, data_fitted2.reshape(xx2.shape), 4, colors='black',linewidths=0.5,linestyles="dashed")
    cb = plt.colorbar(cax)
    cb.set_label(r'$I_{\nu}\, / \,a.u.$')
    plt.xlabel(r'$x\, /\, \mu m$')
    plt.ylabel(r'$y\, /\, \mu m$')
    plt.tight_layout()
    #plt.savefig(savedir + "gaussfit/" + ids[i] + "_difference.png")
    plt.savefig(savedir + "gaussfit/" + ids[i] + "_difference.eps",dpi=1200)
    plt.savefig(savedir + "gaussfit/" + ids[i] + "_difference.pgf")
    plt.close()
    perr = np.sqrt(np.diag(pcov))
    ramg[i] = popt[0]
    stddev[i] = perr[0]
    # plot profiles

    xpos = np.int(round(popt[1]/0.2))
    ypos = np.int(round(popt[2]/0.2))
    fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
    ax0, ax1 = axes
    ax0.plot(yvalues,sub[:,xpos-xstart],'o')
    xd = np.linspace(min(yvalues),max(yvalues),300)
    yd = twoD_Gaussian((np.repeat(popt[1],300),xd), *popt)
    #yd = twoD_Lorentz((np.repeat(popt[1],300),xd), *popt)
    #yd = twoD_Gaussiansqr((np.repeat(popt[1],300),xd), *popt)
    ax0.plot(xd,yd,'-')
    ax0.set_xlabel(r'$x\, /\, \mu m$')
    ax0.set_ylabel(r'$I_{\nu}\, / \,a.u.$')
    ax0.set_title('X direction')
    ax1.plot(xvalues,sub[ypos-ystart,:],'o')
    xd = np.linspace(min(xvalues),max(xvalues),300)
    yd = twoD_Gaussian((xd,np.repeat(popt[2],300)), *popt)
    #yd = twoD_Lorentz((xd,np.repeat(popt[2],300)), *popt)
    #yd = twoD_Gaussiansqr((xd,np.repeat(popt[2],300)), *popt)
    ax1.plot(xd,yd,'-')
    ax1.set_xlabel(r'$y\, /\, \mu m$')
    #ax1.set_ylabel(r'$y\, /\, \mu m$')
    ax1.set_title('Y direction')
    #fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    #plt.show()
    plt.tight_layout()
    #plt.savefig(savedir + "gaussfit/" + ids[i] + "_profile.png", format='png')
    plt.savefig(savedir + "gaussfit/" + ids[i] + "_profile.eps",dpi=1200)
    plt.savefig(savedir + "gaussfit/" + ids[i] + "_profile.pgf")


    plt.close()


#xd = np.linspace(-100,100,500)
#yd = np.linspace(-100,100,500)
#xxd,yyd = np.meshgrid(xd, yd)
#zd = twoD_Lorentz((xxd,yyd),1,100,100,0.1,0.1,0,0)
#fig, ax = plt.subplots(1, 1)
#ax.imshow(zd.reshape(500, 500), cmap=plt.cm.jet)
#plt.show()


fig, ax = plt.subplots()
cax = ax.imshow(data,cmap=cmap,extent=data_extent, origin='lower', interpolation='nearest')
cb = plt.colorbar(cax)
cb.set_label(r'$I_{\nu}\, / \,a.u.$')
for x, y, s in zip(xy[:, 0], xy[:, 1], np.round(ram, 1)):
    plt.text(x - 0.5, y - 1.0, str(s),fontsize=8)
ax.set_xlabel(r'$x\, /\, \mu m$')
ax.set_ylabel(r'$y\, /\, \mu m$')
plt.tight_layout()
#plt.savefig(savedir + "ramanmap_max.png", dpi=300)
#plt.savefig(savedir + "ramanmap_max.pgf")
plt.savefig(savedir + "ramanmap_max.eps",dpi=1200)
plt.savefig(savedir + "ramanmap_max.pgf")
plt.close()

fig, ax = plt.subplots()
cax = ax.imshow(data,cmap=cmap,extent=data_extent, origin='lower', interpolation='nearest')
cb = plt.colorbar(cax)
cb.set_label(r'$I_{\nu}\, / \,a.u.$')
for x, y, s in zip(xy[:, 0], xy[:, 1], np.round(ramg, 1)):
    plt.text(x - 0.5, y + 1.0, str(s),fontsize=8)
ax.set_xlabel(r'$x\, /\, \mu m$')
ax.set_ylabel(r'$y\, /\, \mu m$')
plt.tight_layout()
#plt.savefig(savedir + "ramanmap_max.png", dpi=300)
#plt.savefig(savedir + "ramanmap_max.pgf")
plt.savefig(savedir + "ramanmap_fit.eps",dpi=1200)
plt.savefig(savedir + "ramanmap_fit.pgf")

plt.close()

fig, ax = plt.subplots()
cax = ax.imshow(data,cmap=cmap,extent=data_extent, origin='lower', interpolation='nearest')
cb = plt.colorbar(cax)
cb.set_label(r'$I_{\nu}\, / \,a.u.$')
ax.set_xlabel(r'$x\, /\, \mu m$')
ax.set_ylabel(r'$y\, /\, \mu m$')
plt.tight_layout()
#plt.savefig(savedir + "ramanmap.png", dpi=300)
#plt.savefig(savedir + "ramanmap.pgf")
plt.savefig(savedir + "ramanmap.eps",dpi=1200)
plt.savefig(savedir + "ramanmap.pgf")
plt.close()



f = open(savedir + "peaks_Raman.csv", 'w')
f.write("x,y,id,max,max_gauss,stddev" + "\r\n")
for i in range(len(ids)):
    f.write(str(xy[i, 0]) + "," + str(xy[i, 1]) + "," + str(ids[i]) + "," + str(ram[i]) + "," + str(ramg[i]) + "," + str(stddev[i]) + "\r\n")

f.close()

