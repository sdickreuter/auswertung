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
path = '/home/sei/MNPBEM/Annika/'


#sims = ['dimer_r45nm_d2nm.mat','dimer_r45nm_d5nm.mat','dimer_r45nm_d10nm.mat']
#sims = ['1nm dist/dimer_r45nm_d5nm.mat','01nm dist/dimer_r45nm_d5nm.mat']

#sims = ['01nm dist/dimer_r45nm_d2nm.mat','01nm dist/dimer_r45nm_d5nm.mat','01nm dist/dimer_r45nm_d10nm.mat','01nm dist/dimer_r45nm_d15nm.mat','01nm dist/dimer_r45nm_d20nm.mat','01nm dist/dimer_r45nm_d25nm.mat','01nm dist/dimer_r45nm_d30nm.mat',]
#sims = ['dimer_r45nm_d1nm.mat','dimer_r45nm_d2nm.mat','dimer_r45nm_d3nm.mat','dimer_r45nm_d4nm.mat','dimer_r45nm_d5nm.mat','dimer_r45nm_d10nm.mat','dimer_r45nm_d15nm.mat','dimer_r45nm_d20nm.mat','dimer_r45nm_d25nm.mat','dimer_r45nm_d30nm.mat','dimer_r45nm_d35nm.mat','dimer_r45nm_d40nm.mat']
#sims = ['dimer_r45nm_d2nm_theta0.mat']
#sims = ['dimer_r45nm_d2nm_theta45.mat','dimer_r45nm_d5nm_theta45.mat','dimer_r45nm_d10nm_theta45.mat','dimer_r45nm_d15nm_theta45.mat','dimer_r45nm_d20nm_theta45.mat','dimer_r45nm_d30nm_theta45.mat','dimer_r45nm_d40nm_theta45.mat']


listdir = os.listdir(path)
sims = []
for file in listdir:
    if re.search(r"(.mat)$", file) is not None:
        sims.append(file)

print(sims)
gaps = np.repeat(0,len(sims))


# sims = ['dimer_45degillu_r45nm_d10nm_theta10_1nm0x_5nmflatxz.mat']

#gaps = []
#for sim in sims:
#    print(re.search('(d)([0-9]{1,2})(nm)', sim).group(2))
#    gaps.append( re.search('(d)([0-9]{1,2})(nm)', sim).group(2)  )
#
#
#gaps = np.array(gaps,dtype=np.int)
#sims = np.array(sims)
#
#sorted = np.argsort(gaps)
#gaps = gaps[sorted]
#sims = sims[sorted]

savedir = path+'plots/'

try:
    os.mkdir(savedir)
except:
    pass


plot_details = True

peakwl_scat = np.zeros(len(gaps),dtype=np.object)
peakwl_charge = np.zeros(len(gaps),dtype=np.object)


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

    p1 = mat['p0']#mat['p1']

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

    pinfty_verts = mat['pinfty']['verts'][0][0]
    pinfty_faces = mat['pinfty']['faces'][0][0][:,0:3]
    pinfty_faces = np.array(pinfty_faces-1,dtype=np.int)
    pinfty_pos = mat['pinfty']['pos'][0][0]

    NA = 0.42
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

    indexes_sca = peakutils.indexes(sca, thres=0.00001, min_dist=2)
    peakwl_scat[n] = wl[indexes_sca]

    charge = np.zeros(len(wl))
    for i in range(len(wl)):
        #charge[i] = np.abs((sig2[i]+sig1[i])/2).max()
        charge[i] = np.abs(np.real(sig2[i])).max()

    indexes_charge = peakutils.indexes(charge, thres=0.1, min_dist=2)
    peakwl_charge[n] = wl[indexes_charge]

    indexes_sca = np.append(indexes_sca,np.argmin(np.abs(wl-520)))

    for ind in indexes_sca:


        #ax1 = plt.subplot('111', projection="polar")
        ax1 = plt.subplot('111')

        x = pinfty_pos[:,0]
        y = pinfty_pos[:,1]
        z = pinfty_pos[:,2]

        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

        val = np.linalg.norm( 0.5 * np.real(np.cross(e[ind], np.conjugate(h[ind]))),axis=1)

        theta2, phi2 = np.meshgrid(np.linspace(0,np.pi/2,200),np.linspace(-np.pi,np.pi,200))
        val2 = interpolate.griddata((theta, phi), val, (theta2, phi2), method='linear', fill_value=0.0)

        theta = np.linspace(0,np.pi/2,200)
        phi = np.linspace(-np.pi,np.pi,200)
        print(val2.shape)

        val3 = np.sum(val2,axis=0)
        # val3 = np.zeros(val2.shape[1])
        # for i in range(val2.shape[1]):
        #     val3[i] = val2[:,i].max()


        polarplot = ax1.plot(theta*180/np.pi,val3)
        ax1.axvline(np.arcsin(0.45)*180/np.pi,linestyle='--')
        ax1.axvline(np.arcsin(0.9)*180/np.pi, linestyle='--')
        ax1.set_xlabel("Abstrahl-Winkel")
        ax1.set_ylabel("gestreute Intensität")
        plt.savefig(savedir + sim[:-4] + "_emissionpattern_at_"+ str(int(round(wl[ind]))) +"nm.png", dpi=400)
        #plt.show()
        plt.close()


