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

def plot_trimesh(verts, faces, val=None, cmap=plt.cm.get_cmap(),
                 vmin=None, vmax=None, **kw):
    """Doesn't work."""
    # raise NotImplementedError

    """Plot a mesh of triangles.

    Input   verts   N x 3 array of vertex locations

            faces   M x 3 array of vertex indices for each face

            val     M list of values for each vertex, used for coloring

            cmap    colormap, defaulting to current colormap

            vmin    lower limit for coloring, defaulting to min(val)

            vmax    upper limit for coloring, defaulting to max(val)

            Other keyword pairs are passed to Poly3DCollection
    """


    v = [[verts[ind, :] for ind in face] for face in faces]

    # if val is not None:
    #     val = np.array(val)  # To be safe - set_array() will expect a numpy array
    #     #norm = colors.Normalize(vmin=val.min(), vmax=val.max(), clip=True)
    #     #mapper = cm.ScalarMappable(norm=norm, cmap=cm.seismic)
    #     colormap = (val-val.min()) / (val.max() - val.min())
    #     colormap = cm.seismic(colormap)
    #
    #     #colormap[:, -1] = 1-val / val.max()
    #     colormap = np.array(colormap, dtype=float)
    #     colormap=colors.to_rgba_array(colormap)

    #poly = Poly3DCollection(v, cmap=cmap, norm=colors.Normalize(clip=True), **kw)
    poly = Poly3DCollection(v, **kw)
    #poly.set_alpha(None)
    # Have to set clip=True in Normalize for when vmax < max(val)
    #poly.set_array(val)  # sets vmin, vmax to min, max of val
    #poly.set_clim(vmin, vmax)  # sets vmin, vmax if not None
    # if val is not None:
    #     poly.set_facecolors(colormap)
    # if val > 1, can't set with facecolor = val in definition.
    # Isn't that bizarre?

    return poly


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arccos(z/r)
    phi     =  np.arctan2(y,x)
    return r,theta,phi

#path = '/home/sei/MNPBEM/10degillu/'
#path = '/home/sei/MNPBEM/new_version/'
#path = '/home/sei/MNPBEM/cone_5nmOx/'
path = '/home/sei/MNPBEM/cone/'

plot_details = True


listdir = os.listdir(path)
sims = []
for file in listdir:
    if re.search(r"(.mat)$", file) is not None:
        sims.append(file)

print(sims)

radius = []
height = []
for sim in sims:
    try:
        print(re.search('(r)([0-9]{1,3})', sim).group(2))
        radius.append( re.search('(r)([0-9]{1,3})', sim).group(2)  )
        print(re.search('(h)([0-9]{1,3})', sim).group(2))
        height.append(re.search('(h)([0-9]{1,3})', sim).group(2))
    except AttributeError as e:
        print(sim+' not a valid simulation file')
        sims.remove(sim)

radius = np.array(radius,dtype=np.int)
height = np.array(height,dtype=np.int)

sims = np.array(sims)

sorted = np.argsort(height)
height = height[sorted]
radius = radius[sorted]
sims = sims[sorted]


savedir = path+'plots/'

try:
    os.mkdir(savedir)
except:
    pass



peakwl_scat = np.zeros(len(sims),dtype=np.object)
peakwl_charge = np.zeros(len(sims),dtype=np.object)


mat = sio.loadmat(path + sims[0])
wl = mat['enei'][0]
img = np.zeros((len(wl),len(sims)))


for n,sim in enumerate(sims):
    print(sim)
    mat = sio.loadmat(path+sim)

    wl = mat['enei'][0]
    sca = np.transpose(mat['sca'])[0]
    img = np.zeros((len(wl), len(sims)))

    img[:,n] = sca
    #wl = wl[:53]
    #sca = sca[:53]

    p0 = mat['p0']

    sig1 = np.zeros(len(wl),dtype=np.object)
    sig2 = np.zeros(len(wl),dtype=np.object)

    for i in range(len(wl)):
        sig = mat['sigs']
        sig1[i] = sig[0,i]['sig1'][0][0].T[0]
        sig2[i] = sig[0,i]['sig2'][0][0].T[0]


    e = np.zeros(len(wl),dtype=np.object)
    h = np.zeros(len(wl),dtype=np.object)

    for i in range(len(wl)):
        e[i] = mat['e'][0,i]
        h[i] = mat['h'][0,i]


    # print(e[20][1:2])
    # print(np.conj(h[20])[1:2])
    # print(np.cross(e[20], np.conj(h[20]))[1:2])

    pinfty_verts = mat['pinfty']['verts'][0][0]
    pinfty_faces = mat['pinfty']['faces'][0][0][:,0:3]
    pinfty_faces = np.array(pinfty_faces-1,dtype=np.int)
    pinfty_pos = mat['pinfty']['pos'][0][0]


    verts1 = p0['verts'][0][0]
    faces1 = p0['faces'][0][0][:,0:3]
    faces1 = np.array(faces1-1,dtype=np.int)
    # print(faces1.shape)
    # print(faces1[0:10,:])

    NA = 0.9
    scat = np.zeros(len(wl))
    for i in range(len(wl)):
        x = pinfty_pos[:,0]
        y = pinfty_pos[:,1]
        z = pinfty_pos[:,2]

        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

        S = np.linalg.norm(0.5 * np.real(np.cross(e[i], np.conjugate(h[i]))), axis=1)
        ind = theta < np.arcsin(NA)
        scat[i] = np.sum(S[ind])

    plt.plot(wl,scat)
    plt.plot(wl,sca)
    indexes_sca = peakutils.indexes(sca, thres=0.001, min_dist=2)
    plt.scatter(wl[indexes_sca],sca[indexes_sca])
    for ind in indexes_sca:
        plt.text(wl[ind],sca[ind],str(int(round(wl[ind]))))
    plt.savefig(savedir + sim[:-4] + "_scattering.png", dpi=400)
    #plt.show()
    plt.close()
    peakwl_scat[n] = wl[indexes_sca]

    plt.plot(wl,scat/scat.max())
    plt.savefig(savedir + sim[:-4] + "_scattering_na.pdf", dpi=400)
    #plt.show()
    plt.close()
    peakwl_scat[n] = wl[indexes_sca]


    charge = np.zeros(len(wl))
    for i in range(len(wl)):
        #charge[i] = np.abs((sig2[i]+sig1[i])/2).max()
        #charge[i] = np.abs(np.real(sig2[i])).max()
        charge[i] = np.abs(np.real(sig2[i])).sum()

    indexes_charge = peakutils.indexes(charge, thres=0.001, min_dist=2)
    plt.scatter(wl[indexes_charge], charge[indexes_charge])
    plt.plot(wl,charge)
    for ind in indexes_charge:
        plt.text(wl[ind],charge[ind],str(int(round(wl[ind]))))
    plt.savefig(savedir + sim[:-4] + "_maxcharge.png", dpi=400)
    #plt.show()
    plt.close()
    peakwl_charge[n] = wl[indexes_charge]

    tipcharge = np.zeros(len(wl))

    tip_indices = []
    for j in range(len(sig2[i])):
        if verts1[faces1[j, :], 2].mean() > (height[n]-20):
            tip_indices.append(j)
    tip_indices = np.array(tip_indices,dtype=np.int)

    fig = plt.figure()
    ax = Axes3D(fig)
    v = [[verts1[ind, :] for ind in face] for face in faces1[tip_indices]]
    poly1 = Poly3DCollection(v)
    poly1.set_facecolor(None)
    poly1.set_edgecolor('blue')
    ax.add_collection3d(poly1)
    ax.set_xlim3d(-20, 20)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(height[n]-30, height[n]+10)
    plt.axis('off')
    plt.savefig(savedir + sim[:-4] + "_tip.png", dpi=400)
    #plt.show()
    plt.close()

    for i in range(len(wl)):
        tipcharge[i] = np.abs(np.real(sig2[i][tip_indices].max()))
    indexes_tipcharge = peakutils.indexes(tipcharge, thres=0.1, min_dist=2)
    plt.scatter(wl[indexes_tipcharge], tipcharge[indexes_tipcharge])
    plt.plot(wl, tipcharge)
    for ind in indexes_tipcharge:
        plt.text(wl[ind],tipcharge[ind],str(int(round(wl[ind]))))
    plt.savefig(savedir + sim[:-4] + "_tipcharge.png", dpi=400)
    #plt.show()
    plt.close()

    if plot_details:
        #for ind in indexes_charge:
        #for ind in indexes_sca:
        #for ind in indexes_tipcharge:
        for ind in np.hstack((indexes_charge,indexes_sca,indexes_tipcharge)):

            val = np.real(sig1[ind])

            fig = plt.figure()

            ax = Axes3D(fig)
            ax.set_aspect('equal')

            v = [[verts1[ind, :] for ind in face] for face in faces1]

            poly1 = Poly3DCollection(v)

            val1 = np.array(val[:len(faces1)])  # To be safe - set_array() will expect a numpy array
            val1_norm = (val1-val1.min()) / (val1.max() - val1.min())
            colormap = cm.seismic(val1_norm)
            colormap[:, -1] = 0.4*signal.sawtooth(2 * np.pi * val1_norm+np.pi,0.5)+0.4
            poly1.set_facecolor(colormap)
            poly1.set_edgecolor(None)

            ax.add_collection3d(poly1)
            #ax.auto_scale_xyz([verts1[:,0].max(),verts1[:,0].min()],[verts1[:,1].max(),verts1[:,1].min()],[verts1[:,2].max(),verts1[:,2].min()])
            #ax.set_xlim3d(verts1[:,0].min(), verts1[:,0].max())
            #ax.set_ylim3d(verts1[:,1].min(), verts1[:,1].max())
            #ax.set_zlim3d(verts1[:,2].min(), verts1[:,2].max())
            ax.set_xlim3d(-(radius.max()+5), radius.max()+5)
            ax.set_ylim3d(-(radius.max()+5), radius.max()+5)
            ax.set_zlim3d(0, height.max()+10)

            plt.axis('off')
            plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm.pdf", dpi=400)
            # plt.show()
            plt.close()


            gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1], )

            ax1 = plt.subplot(gs[0], projection="polar", aspect=1.)
            ax2 = plt.subplot(gs[1])

            x = pinfty_pos[:,0]
            y = pinfty_pos[:,1]
            z = pinfty_pos[:,2]

            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)

            val = np.linalg.norm( 0.5 * np.real(np.cross(e[ind], np.conjugate(h[ind]))),axis=1)

            theta2, phi2 = np.meshgrid(np.linspace(0,np.pi/2,500),np.linspace(-np.pi,np.pi,500))
            val2 = interpolate.griddata((theta, phi), val, (theta2, phi2), method='nearest', fill_value=0.0)
            polarplot = ax1.pcolormesh(phi2,theta2, val2)
            ax1.set_rticks([0.5, 1, 1.5])
            ax1.set_rlabel_position(90)
            plt.colorbar(polarplot, cax=ax2)
            plt.savefig(savedir + sim[:-4] + "_farfield_at_"+ str(int(round(wl[ind]))) +"nm.png", dpi=400)
            #plt.show()
            plt.close()

