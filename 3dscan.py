import numpy as np
import re

#from plotsettings import *

#import matplotlib
#matplotlib.use('QT4Agg')

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib import pyplot as plt
#from seaborn import cubehelix_palette
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
#from mayavi import mlab
#from pyevtk.hl import pointsToVTK
import pandas as pd
import pyvtk
from scipy.spatial import Delaunay
from scipy.signal import savgol_filter
from skimage import morphology
from scipy import ndimage
import os
from scipy.optimize import minimize, basinhopping
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
#path = '/home/sei/Spektren/p45m_did5_par_A1_scan3D_2/'
path = '/home/sei/Spektren/p45m_did5_par_A1_scan3D_newlense/'
#path = '/home/sei/Spektren/cone3dnikon5/'


filename = path+'cube.csv'

maxwl = 1000
minwl = 450

def gauss3d(x, y, z, amplitude, xo, yo, zo, fwhmx, fwhmy, fwhmz):
    sigmax = fwhmx / 2.3548
    sigmay = fwhmy / 2.3548
    sigmaz = fwhmz / 2.3548
    g = amplitude*np.exp(-np.square(x - xo)/(2 * np.square(sigmax))-np.square(y - yo)/ (2 * np.square(sigmay))-np.square(z - zo)/ (2 * np.square(sigmaz)) )
    return g.ravel()
#
# x = np.linspace(-1,1,25)
# y = np.linspace(-1,1,25)
# z = np.linspace(-1,1,25)
# x,y,z = np.meshgrid(x,y,z)
# x = x.ravel()
# y = y.ravel()
# z = z.ravel()
# r = np.sqrt(np.square(x)+np.square(y)+np.square(z))
# r -= r.min()
# r /= r.max()
# cm = plt.cm.get_cmap('viridis')
# colors = cm(r)
# mag = gauss3d(x, 0, y, 0, z, 0, 0.5, 1.0, 0.1)
# mag -= mag.min()
# mag /= mag.max()
# size = mag*50+1
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x,y,z,c=colors,s=size,alpha=0.2,lw = 0)
# plt.show()

d = pd.read_csv(filename, delimiter=',')
wl, lamp = np.loadtxt(open(path + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, dark = np.loadtxt(open(path + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, bg = np.loadtxt(open(path + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)


x = d.x.values
y = d.y.values
z = d.z.values
wl = np.array(list(d)[3:-1],dtype=np.float)
mask = (wl >= minwl) & (wl <= maxwl)

d = d.values
d = d[:,3:-1]

for i in range(d.shape[0]):
    #d[i,:] = savgol_filter((d[i,:]-bg)/(lamp-dark),51,0)
    d[i,:] = savgol_filter(d[i,:],51,0)

d = d[:, mask]
wl = wl[mask]
print(d.shape)

# x=-x
# y=-y

x = x - x.min()
y = y - y.min()
z = z - z.min()

n = int(np.round(np.power(len(x),1/3)))
print(n)
nx = np.linspace(0,x.max(),n)
ny = np.linspace(0,y.max(),n)
nz = np.linspace(0,z.max(),n)

gridx, gridy, gridz = np.meshgrid(nx,ny,nz)

int_pos = np.zeros(wl.shape)
int_max = np.zeros(wl.shape)

grid = np.zeros((len(wl),n,n,n))
fgrid = np.zeros((len(wl),n,n,n))

for i,wavelength in enumerate(wl):
    grid[i,:,:,:] = griddata((x, y, z), d[:, i], (gridx, gridy, gridz), method='nearest')
    #fgrid =  gaussian_filter(grid,1)#gaussian_filter(grid,3)
    fgrid[i,:,:,] = ndimage.median_filter( grid[i,:,:,:], 3)


fgrid -= fgrid.min()
fgrid /= fgrid.max()

from pyevtk.hl import gridToVTK
from pyevtk.vtk import VtkGroup

try:
    os.mkdir(path + "3d/")
except:
    pass


wavelengths = np.arange(400,910,10) #nm
os.chdir(path+'3d/')
g = VtkGroup("3dscan")
#fileNames = []
for wavelength in wavelengths:
    wl_ind = np.argmin(np.abs(wl - wavelength))
    #gridToVTK("3d/3dscan"+str(round(wavelength)), x, y, z, cellData = {"intensity" : fgrid[wl_ind,:,:,]}, pointData = {"intensityp" : fgrid[wl_ind,:,:,]})
    gridToVTK("3dscan" + str(round(wavelength)), nx.ravel(), ny.ravel(), nz.ravel(),pointData={"intensityp": fgrid[wl_ind, :, :, ]})
    #gridToVTK("3d/3dscan"+str(round(wavelength)), x.ravel(), y.ravel(), z.ravel(), cellData = {"intensity" : fgrid[wl_ind,:,:,].ravel()})
    g.addFile(filepath = "./3dscan"+str(round(wavelength))+'.vtr', sim_time = round(wavelength))
    #fileNames.append("3dscan" + str(round(wavelength)))

g.save()
os.chdir('..')


# import vtk_export
#
# vtk_writer = vtk_export.VTK_XML_Serial_Structured()
# wavelengths = np.arange(400,900,10) #nm
# os.chdir('3d/')
# for wavelength in wavelengths:
#     wl_ind = np.argmin(np.abs(wl - wavelength))
#     vtk_writer.snapshot("3dscan"+str(round(wavelength))+".vtu", gridx.ravel(), gridy.ravel(), gridz.ravel(), intensity=fgrid[wl_ind,:,:,].ravel())
# vtk_writer.writePVD("3dscan.pvd")
# os.chdir('..')


# fgrid = fgrid[:,5:-3,5:-3,:]
# fgrid -= fgrid.min()
#
# nx = np.linspace(0,x.max(),fgrid.shape[1])
# ny = np.linspace(0,y.max(),fgrid.shape[2])
# nz = np.linspace(0,z.max(),fgrid.shape[3])
# gridx, gridy, gridz = np.meshgrid(nx,ny,nz)


# from mayavi import mlab
# mlab.figure(bgcolor=(0, 0, 0))
# #mlab.figure(1, bgcolor=(0, 0, 0), size=(350, 350))
# #mlab.clf()
#
# s = mlab.pipeline.scalar_field(fgrid[300,:,:,:])
# # contour = mlab.pipeline.contour(s)
# #smooth = mlab.pipeline.user_defined(contour, filter='SmoothPolyDataFilter')
# #smooth.filter.number_of_iterations = 400
# #smooth.filter.relaxation_factor = 0.015
# # curv = mlab.pipeline.user_defined(smooth, filter='Curvatures')
# # surf = mlab.pipeline.surface(curv)
# # module_manager = curv.children[0]
# # module_manager.scalar_lut_manager.data_range = np.array([-0.6,  0.5])
# # module_manager.scalar_lut_manager.lut_mode = 'RdBu'
#
#
# # min_fgrid = int(fgrid[300,:,:,:].min())
# # max_fgrid = int(fgrid[300,:,:,:].max())
# # print(min_fgrid)
# # print(max_fgrid)
# # #vol = mlab.pipeline.volume(s, vmin=min_fgrid + 0.5 * (max_fgrid - min_fgrid), vmax=min_fgrid + 0.9 * (max_fgrid - min_fgrid))
# vol = mlab.pipeline.volume(s)
#
#
#
# mlab.show()


# def plotCubeAt(pos=(0,0,0),ax=None,color='b'):
#     # Plotting a cube element at position pos
#     if ax !=None:
#         #X, Y, Z = cuboid_data( pos )
#         #ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=1)
#         verts = np.array([[1.0, 1.0, -1.0],
#                  [1.0, -1.0, -1.0],
#                  [-1.0, -1.0, -1.0],
#                  [-1.0, 1.0, -1.0],
#                  [1.0, 1.0, 1.0],
#                  [1.0, -1.0, 1.0],
#                  [-1.0, -1.0, 1.0],
#                  [-1.0, 1.0, 1.0]])
#         verts[:,0] += pos[0]
#         verts[:, 1] += pos[1]
#         verts[:, 2] += pos[2]
#         faces = np.array([[0, 1, 2, 3],
#                  [4, 7, 6, 5],
#                  [0, 4, 5, 1],
#                  [1, 5, 6, 2],
#                  [2, 6, 7, 3],
#                  [4, 0, 3, 7]],dtype=np.int)
#         v = [[verts[ind, :] for ind in face] for face in faces]
#         #ax.add_collection3d(Poly3DCollection(v, facecolors=color, linewidths=1, edgecolors='r', alpha=.25))
#         ax.add_collection3d(Poly3DCollection(v, facecolors=color, linewidths=1, edgecolors=None))
#
#
# def plotMatrix(ax, matrix,colors=None):
#     # plot a Matrix
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             for k in range(matrix.shape[2]):
#                 if matrix[i,j,k] == 1:
#                     # to have the
#                     if colors is not None:
#                         plotCubeAt(pos=(i-0.5,j-0.5,k-0.5), ax=ax,color=colors[i,j,k,:])
#                     else:
#                         plotCubeAt(pos=(i-0.5,j-0.5,k-0.5), ax=ax)
#
# cmap = plt.get_cmap('plasma')
#
# wl_ind = np.argmin(np.abs(wl-600))
#
# filled = fgrid[wl_ind,:,:,:] > fgrid[wl_ind,:,:,:].max()*0.2
# filled = filled.reshape(fgrid[wl_ind,:,:,:].shape)
#
# cdata = fgrid[wl_ind,:,:,:].ravel()
# colors = np.zeros( (len(cdata),4) )
# cdata -= cdata[filled.ravel()].min()
# cdata /= cdata[filled.ravel()].max()
# for i in range(len(cdata)):
#     colors[i,:] = cmap(cdata[i])
#     if cdata[i] > 0:
#         colors[i,3] = (cdata[i]+0.2)/1.2
#     else:
#         colors[i,3] = 0
#
# colors = colors.reshape(fgrid[wl_ind,:,:,:].shape+(4,))
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect('equal')
# ax.axis('equal')
#
# ax.auto_scale_xyz([0, 23], [0, 23], [0, 23])
#
# plotMatrix(ax, filled,colors=colors)
#
# plt.show()
#



#
#
#
# def getCube(pos=(0,0,0)):
#     # Plotting a cube element at position pos
#     #X, Y, Z = cuboid_data( pos )
#     #ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=1)
#     verts = np.array([[1.0, 1.0, -1.0],
#              [1.0, -1.0, -1.0],
#              [-1.0, -1.0, -1.0],
#              [-1.0, 1.0, -1.0],
#              [1.0, 1.0, 1.0],
#              [1.0, -1.0, 1.0],
#              [-1.0, -1.0, 1.0],
#              [-1.0, 1.0, 1.0]])
#     verts[:,0] += pos[0]
#     verts[:, 1] += pos[1]
#     verts[:, 2] += pos[2]
#     faces = np.array([[0, 1, 2, 3],
#              [4, 7, 6, 5],
#              [0, 4, 5, 1],
#              [1, 5, 6, 2],
#              [2, 6, 7, 3],
#              [4, 0, 3, 7]],dtype=np.int)
#     #ax.add_collection3d(Poly3DCollection(v, facecolors=color, linewidths=1, edgecolors='r', alpha=.25))
#     return verts, faces
#
# def plotMatrix(ax, matrix,colors=None):
#     # plot a Matrix
#     colors2 = np.zeros((0,4))
#     verts = np.zeros((0,3))
#     faces = np.zeros((0,4),dtype=np.int)
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             for k in range(matrix.shape[2]):
#                 if matrix[i,j,k] == 1:
#                     v, f = getCube(pos=(i - 0.5, j - 0.5, k - 0.5))
#                     verts = np.vstack( (verts,v) )
#                     faces = np.vstack( (faces,f) )
#                     for l in range(len(faces)):
#                         colors2 = np.vstack( (colors2,colors[i,j,k,:]) )
#     v = [[verts[ind, :] for ind in face] for face in faces]
#     v = np.array(v)
#     print(colors.shape)
#     print(v.shape)
#     print(colors2.shape)
#     ax.add_collection3d(Poly3DCollection(verts, facecolors=colors2, linewidths=1, edgecolors=None))
#     #ax.add_collection3d(Poly3DCollection(v, linewidths=1, edgecolors=None))
#
#
# cmap = plt.get_cmap('plasma')
#
# wl_ind = np.argmin(np.abs(wl-600))
#
# filled = fgrid[wl_ind,:,:,:] > fgrid[wl_ind,:,:,:].max()*0.1
# filled = filled.reshape(fgrid[wl_ind,:,:,:].shape)
#
# cdata = fgrid[wl_ind,:,:,:].ravel()
# colors = np.zeros( (len(cdata),4) )
# cdata -= cdata[filled.ravel()].min()
# cdata /= cdata[filled.ravel()].max()
# for i in range(len(cdata)):
#     colors[i,:] = cmap(cdata[i])
#     if cdata[i] > 0:
#         colors[i,3] = (cdata[i]+0.2)/1.2
#     else:
#         colors[i,3] = 0
#
# colors = colors.reshape(fgrid[wl_ind,:,:,:].shape+(4,))
# #colors = colors.reshape((144,4,3,4))
# print(colors.shape)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect('equal')
# ax.axis('equal')
#
# ax.auto_scale_xyz([0, 23], [0, 23], [0, 23])
#
# plotMatrix(ax, filled,colors=colors)
#
# plt.show()
#
#
#



#
#
# # combine the color components
# cmap = plt.get_cmap('plasma')
#
# wl_ind = np.argmin(np.abs(wl-600))
#
# filled = fgrid[wl_ind,:,:,:] > fgrid[wl_ind,:,:,:].max()*0.2
# filled = filled.reshape(fgrid[wl_ind,:,:,:].shape)
#
# cdata = fgrid[wl_ind,:,:,:].ravel()
# colors = np.zeros( (len(cdata),4) )
# cdata -= cdata[filled.ravel()].min()
# cdata /= cdata[filled.ravel()].max()
# for i in range(len(cdata)):
#     colors[i,:] = cmap(cdata[i])
#     if cdata[i] > 0:
#         colors[i,3] = (cdata[i]+0.2)/1.2
#     else:
#         colors[i,3] = 0
#
# #ind = cdata < 0
# #colors[ind,:] = [0,0,0,0]
# colors = colors.reshape(fgrid[wl_ind,:,:,:].shape+(4,))
#
# # and plot everything
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# #ax.voxels(gridx, gridy, gridz, filled = filled,
# ax.voxels(filled = filled,
#           facecolors=colors,
#           edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
#           #edgecolors=None,
#           linewidth=0.5)
# ax.set(xlabel='x', ylabel='y', zlabel='z')
#
# plt.show()
#
#
#
# for i, wavelength in enumerate(wl):
#     int_pos[i] = np.max(fgrid[i,int(len(nx)/2),int(len(nx)/2),int(len(nx)/2)])
#     int_max[i] = np.max(fgrid[i,:,:,:])
#
#
# int_pos -= int_pos.min()
# int_pos /= int_pos.max()
# int_max -= int_max.min()
# int_max /= int_max.max()
#
# plt.plot(wl,int_pos)
# plt.plot(wl,int_max)
# plt.legend(("at middle","max values"))
# plt.savefig(path+"compare.png",dpi=300)
#
# print("Compare plot saved")
#
# try:
#     os.mkdir(path + "focus_plots/")
# except:
#     pass
#
# wavelengths = np.arange(400,900,2) #nm
# xi = np.array([])
# yi = np.array([])
# zi = np.array([])
# intensity = np.array([])
#
# plot_wavelengths = np.arange(400,1000,10) #nm
#

# for i,wavelength in enumerate(wavelengths):
#     wl_ind = np.abs(np.subtract(wl, wavelength)).argmin()
#     #grid = griddata((x, y, z), d[:, wl_ind], (gridx, gridy, gridz), method='nearest')
#     #fgrid =  gaussian_filter(grid,1)#gaussian_filter(grid,3)
#     #fgrid = ndimage.median_filter(grid, 5)
#     #fgrid -= fgrid.min()
#     xind, yind, zind = np.unravel_index(fgrid[i,:,:,:].argmax(), fgrid[i,:,:,:].shape)
#
#     intensity = np.append(intensity,fgrid[i,:,:,:].max())
#     xi = np.append(xi,xind)
#     yi = np.append(yi,yind)
#     zi = np.append(zi,zind)
#
#     def err_fun(p):
#         #fit = gauss3d(x,y,z, *p)
#         fit = gauss3d(gridx,gridy,gridz, *p)
#         diff = np.abs(fgrid[i,:,:,:].ravel() - fit) ** 2
#         return np.sum(diff)
#
#     initial_guess = (1e-3,1.0,1.0,1.0,1.0,1.0,1.0)
#     minimizer_kwargs = {"method": "SLSQP", "tol": 1e-12}
#     res = basinhopping(err_fun, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=50)
#     popt = res.x
#     print(popt)
#
#     intensity = np.append(intensity,popt[0])
#     xi = np.append(xi,popt[1])
#     yi = np.append(yi,popt[2])
#     zi = np.append(zi,popt[3])
#
#
#     if fgrid[i,xind,yind,zind] > np.std(grid.ravel()):
#         intensity = np.append(intensity,fgrid[i,xind,yind,zind])
#         xi = np.append(xi,nx[xind])
#         yi = np.append(yi,ny[yind])
#         zi = np.append(zi,nz[zind])
#
#
#     if wavelength in plot_wavelengths:
#         fig = plt.figure()
#         plt.imshow(fgrid[wavelength,:,:,zind], extent=(nx.min(),nx.max(),ny.min(),ny.max()) )
#         plt.xlabel("x / um")
#         plt.ylabel("y / um")
#         plt.savefig(path+"focus_plots/intensity_" + str(round(wavelength)) +"nm_z_"+str(zind)+".png", dpi= 300)
#         plt.close()


xi = np.array([])
yi = np.array([])
zi = np.array([])
intensity = np.array([])


for i,wavelength in enumerate(wl):
    xind, yind, zind = np.unravel_index(fgrid[i,:,:,:].argmax(), fgrid[i,:,:,:].shape)

    intensity = np.append(intensity,fgrid[i,:,:,:].max())
    xi = np.append(xi,xind)
    yi = np.append(yi,yind)
    zi = np.append(zi,zind)


mask = intensity > intensity.mean()/2#/2
zi = zi[mask]
intensity=intensity[mask]
wls = wl[mask]

# fig = plt.figure()
# ax = fig.add_subplot(111)
fig, ax = plt.subplots()

#divider = make_axes_locatable(ax)
#cax = divider.append_axes('right', size='5%', pad=0.05)

int_norm = intensity
int_norm = int_norm-int_norm.min()
int_norm = int_norm/int_norm.max()
cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in int_norm]
# cmap = plt.get_cmap('rainbow')
# colors = [cmap(i) for i in np.linspace(0, 1, len(wls))]

#ax.scatter(zi,wls,c=colors,s=(int_norm*50)+1)
ax.scatter(wls,zi,c=colors)
ax.set_ylabel(r'$z\, /\, \mu m$')
ax.set_xlabel(r'$\lambda\, /\, nm$')

# divider = make_axes_locatable(ax)
# m = plt.cm.ScalarMappable(cmap=cmap)
# m.set_array(wls)
# cb = fig.colorbar(m, orientation='vertical')
# #cb = fig.colorbar(m, cax=cax, orientation='vertical')
# #cb = plt.colorbar(m, cax=ax, orientation='vertical')
# #cb = plt.colorbar(m, cax=ax, orientation='horizontal')
# tick_locator = ticker.MaxNLocator(nbins=5)
# cb.locator = tick_locator
# cb.update_ticks()
# cb.ax.tick_params(axis='y', direction='out')
# cb.set_label(r'$\lambda\, /\, nm$')

divider = make_axes_locatable(ax)
m = plt.cm.ScalarMappable(cmap=cmap)
m.set_array(int_norm)
cb = fig.colorbar(m, orientation='vertical')
#tick_locator = ticker.MaxNLocator(nbins=5)
#cb.locator = tick_locator
#cb.update_ticks()
cb.ax.tick_params(axis='y', direction='out')
cb.set_label(r'$I_{df}\, /\, a.u.$')


plt.tight_layout()
plt.savefig(path + "maximum_shift.png", dpi=300)
plt.savefig(path + "maximum_shift.pgf")
plt.savefig(path + "maximum_shift.pdf")

#plt.show()

# intensity -= intensity.min()
# intensity /= intensity.max()
# size = intensity*200
# cm = plt.cm.get_cmap('rainbow')
# colors = cm(np.linspace(0, 1, len(intensity)))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xi,yi,zi,c=colors,s=size,alpha=0.2,lw = 0)
# max_range = np.array([nx.max()-nx.min(), ny.max()-ny.min(), nz.max()-nz.min()]).max() / 2.0
# mid_x = (nx.max()+nx.min()) * 0.5
# mid_y = (ny.max()+ny.min()) * 0.5
# mid_z = (nz.max()+nz.min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)
# ax.set_xlabel('x / um')
# ax.set_ylabel('y / um')
# ax.set_zlabel('z / um')
# plt.show()


#
# wavelengths = np.arange(400,1000,20) #nm
#
# try:
#     os.mkdir(path + "xy_plots/")
# except:
#     pass
#
# for wavelength in wavelengths:
#
#     wl_ind = np.abs(np.subtract(wl,wavelength)).argmin()
#     #grid = griddata((x, y, z), d[:, wl_ind], (gridx, gridy, gridz), method='nearest')
#     #fgrid = ndimage.median_filter(grid, 3)
#     plt.imshow(fgrid[wavelength,:,:,int(len(nz)/2)], extent=(nx.min(),nx.max(),ny.min(),ny.max()) )
#     plt.xlabel("x / um")
#     plt.ylabel("y / um")
#     #plt.show()
#     plt.savefig(path+"xy_plots/intensity_" + str(round(wavelength)) +"nm.png", dpi= 300)
#
#
#
# try:
#     os.mkdir(path + "zy_plots/")
# except:
#     pass
#
# for wavelength in wavelengths:
#
#     wl_ind = np.abs(np.subtract(wl,wavelength)).argmin()
#     #grid = griddata((x, y, z), d[:, wl_ind], (gridx, gridy, gridz), method='nearest')
#     #fgrid = ndimage.median_filter(grid, 3)
#     plt.imshow(fgrid[wavelength,:,int(len(ny)/2),:], extent=(nx.min(),nx.max(),nz.min(),nz.max()) )
#     plt.xlabel("x / um")
#     plt.ylabel("z / um")
#     #plt.show()
#     plt.savefig(path+"zy_plots/intensity_" + str(round(wavelength)) +"nm.png", dpi= 300)
#
#
# try:
#     os.mkdir(path + "zx_plots/")
# except:
#     pass
#
# for wavelength in wavelengths:
#
#     wl_ind = np.abs(np.subtract(wl,wavelength)).argmin()
#     #grid = griddata((x, y, z), d[:, wl_ind], (gridx, gridy, gridz), method='nearest')
#     #fgrid = ndimage.median_filter(grid, 3)
#     plt.imshow(fgrid[wavelength,int(len(nx)/2),:,:], extent=(ny.min(),ny.max(),nz.min(),nz.max()) )
#     plt.xlabel("y / um")
#     plt.ylabel("z / um")
#     #plt.show()
#     plt.savefig(path+"zx_plots/intensity_" + str(round(wavelength)) +"nm.png", dpi= 300)
#

















# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x,y,z,c=d[:,wl_ind],s=20.0,alpha=0.5)
# plt.show()







#pointsToVTK(path+"cube.vtk", x, y, z, data = dict((wl[i],d[:,i]) for i in range((d.shape[1]))))

#pointsToVTK(path+"cube.vtk", x, y, z, data = {"temp" : temp, "pressure" : pressure})

# with open(filename, 'rw') as f:
#     linecount = 0
#     for line in f:
#         if linecount < 1:
#
#
#         print(line, end='')
