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
path = '/home/sei/MNPBEM/cone_chrissi/'


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


    # print(e[20][1:2])
    # print(np.conj(h[20])[1:2])
    # print(np.cross(e[20], np.conj(h[20]))[1:2])

    pinfty_verts = mat['pinfty']['verts'][0][0]
    pinfty_faces = mat['pinfty']['faces'][0][0][:,0:3]
    pinfty_faces = np.array(pinfty_faces-1,dtype=np.int)
    pinfty_pos = mat['pinfty']['pos'][0][0]


    #verts = np.vstack( (p1['verts'][0][0],p2['verts'][0][0]))
    #faces = np.vstack( (p1['faces'][0][0],p2['faces'][0][0]))[:,0:3]
    verts1 = p1['verts'][0][0]
    faces1 = p1['faces'][0][0][:,0:3]
    faces1 = np.array(faces1-1,dtype=np.int)

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

    charge = np.zeros(len(wl))
    for i in range(len(wl)):
        #charge[i] = np.abs((sig2[i]+sig1[i])/2).max()
        charge[i] = np.abs(np.real(sig2[i])).max()

    indexes_charge = peakutils.indexes(charge, thres=0.1, min_dist=2)
    plt.scatter(wl[indexes_charge], charge[indexes_charge])
    plt.plot(wl,charge)
    for ind in indexes_charge:
        plt.text(wl[ind],charge[ind],str(int(round(wl[ind]))))
    plt.savefig(savedir + sim[:-4] + "_maxcharge.png", dpi=400)
    #plt.show()
    plt.close()
    peakwl_charge[n] = wl[indexes_charge]


    #wls = [525,584,694]
    #sca_buf = sca
    #sca[wl < 500] = 0
    #wls = [wl[np.argmax([sca])]]

    if plot_details:
        for ind in indexes_charge:
            #ind = np.abs(wl - w).argmin()

            #val = np.real((sig2[ind]+sig1[ind])/2)
            val = np.real(sig1[ind])

            fig = plt.figure()
            #fig = plt.figure(figsize=plt.figaspect(1) * 1.5)  # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
            #ax = fig.gca(projection='3d')
            ax = Axes3D(fig)
            #ax.set_aspect('equal')
            ax.axis('equal')

            poly1 = plot_trimesh(verts1,faces1,val[:len(faces1)])
            val1 = np.array(val[:len(faces1)])  # To be safe - set_array() will expect a numpy array
            val1_norm = (val1-val1.min()) / (val1.max() - val1.min())
            colormap = cm.seismic(val1_norm)
            colormap[:, -1] = 0.499*signal.sawtooth(2 * np.pi * val1_norm,0.5)+0.501
            poly1.set_facecolor(colormap)
            poly1.set_edgecolor(None)

            ax.add_collection3d(poly1)

            verts = verts1
            #ax.auto_scale_xyz([verts[:,0].max(),verts[:,0].min()],[verts[:,1].max(),verts[:,1].min()],[verts[:,2].max(),verts[:,2].min()])
            #ax.set_xlim3d(verts[:,0].min(), verts[:,0].max())
            #ax.set_ylim3d(verts[:,1].min(), verts[:,1].max())
            #ax.set_zlim3d(verts[:,2].min(), verts[:,2].max())
            #ax.auto_scale_xyz(np.vstack((verts1[:, 0],verts2[:, 0])), np.vstack((verts1[:, 1],verts2[:, 1])), np.vstack((verts1[:, 2],verts2[:, 2])))
            ax.axis('equal')
            #ax.auto_scale_xyz(np.vstack((verts1[:, 0],verts2[:, 0])), np.vstack((verts1[:, 1],verts2[:, 1])), np.vstack((verts1[:, 2],verts2[:, 2])))
            ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
            #plt.title('wl: '+str(round(wl[ind]))+'  max: '+str(round(val.max(),2)))
            plt.axis('off')
            plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm.png", dpi=400)
            #plt.show()
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


fig, ax1 = plt.subplots()
for i in range(len(gaps)):
    for peakwl in peakwl_scat[i]:
        ax1.scatter(gaps[i],peakwl,color='C0')

for i in range(len(gaps)):
    for peakwl in peakwl_charge[i]:
        ax1.scatter(gaps[i],peakwl,color='C1')

ax1.set_ylabel("Peak Wavelength / nm")
ax1.set_xlabel("Gap Width / nm")
plt.savefig(savedir + sim[:-4] + "_spectral_shift.png", dpi=400)
#plt.show()
plt.close()

img = img.T
#newfig(0.9)
fig = plt.figure()
plt.imshow(img, aspect='auto', cmap=plt.get_cmap("viridis"), extent=[wl.min(), wl.max(), 0, len(sims)], norm=colors.LogNorm())
plt.ylabel(r'$number of measurement$')
plt.xlabel(r'$\lambda\, /\, nm$')
#plt.savefig(savedir + sim[:-4] + "_image_log.pdf")
plt.savefig(savedir + sim[:-4] + "_image_log.png", dpi=400)
# plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
plt.close()

#newfig(0.9)
fig = plt.figure()
plt.imshow(img, aspect='auto', cmap=plt.get_cmap("viridis"), extent=[wl.min(), wl.max(), 0, len(sims)])
plt.ylabel(r'$number of measurement$')
plt.xlabel(r'$\lambda\, /\, nm$')
#plt.savefig(savedir + sim[:-4] + "_image.pdf")
plt.savefig(savedir + sim[:-4] + "_image.png", dpi=400)
# plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
plt.close()





# midpoints = np.zeros((len(faces1),3))
    # for f in range(len(faces1)):
    #     midpoints[f] = np.sum(verts1[faces1[f,:],:],axis=0)/3
    #
    # midpoints -= np.mean(midpoints,axis=0)
    #
    # theta = np.zeros((len(faces1)))
    # phi = np.zeros((len(faces1)))
    # for i in range(len(midpoints)):
    #     r, theta[i], phi[i] = asSpherical(midpoints[i])
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    # ax.scatter(midpoints[:,0],midpoints[:,1],midpoints[:,2])
    #
    # pts = np.zeros((len(theta),2))
    # pts[:,0] = theta
    # pts[:,1] = phi
    # tri = Delaunay(pts)
    # centers = np.sum(pts[tri.simplices], axis=1, dtype='int')/3.0
    # print(centers.shape)
    # #plt.tripcolor(pts[:,0], pts[:,1], tri.simplices.copy(), facecolors=val[:len(faces1)], edgecolors='k')
    # plt.tripcolor(pts[:,0], pts[:,1], tri.simplices.copy())
    # plt.gca().set_aspect('equal')
    # plt.show()


    # plt.show()
    # plt.scatter(theta,phi,val[:len(faces1)])
    # plt.show()


    # verts1 -= np.mean(verts1,axis=0)
    # theta = np.zeros((len(verts1)))
    # phi = np.zeros((len(verts1)))
    # for i in range(len(verts1)):
    #     r, theta[i], phi[i] = asSpherical(verts1[i])
    #
    # plt.tripcolor(theta, phi, faces1, facecolors=val[:len(faces1)], edgecolors='k')
    # plt.show()
