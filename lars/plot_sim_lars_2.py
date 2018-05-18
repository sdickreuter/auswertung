import os
import re
import sys

import numpy as np
from scipy.optimize import basinhopping, curve_fit
from scipy.signal import savgol_filter

from plotsettings import *

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

def lorentz(x, amplitude, xo, sigma):
    g = amplitude * np.power(sigma / 2, 2) / (np.power(sigma / 2, 2) + np.power(x - xo, 2))
    return g.ravel()

# https://www.webpages.uidaho.edu/brauns/vibspect1.pdf
def asymlorentz(x, amplitude, x0, sigma, asy):
    sigma = 2 * sigma/(1 + np.exp(asy*(x-x0)) )
    g = lorentz(x,amplitude,x0,sigma)
    return g.ravel()

def three_lorentz(x,c, a0,a1,a2, xo0, xo1, xo2, fwhm0, fwhm1, fwhm2):
    g = lorentz(x,a0,xo0,fwhm0)+lorentz(x,a1,xo1,fwhm1)+lorentz(x,a2,xo2,fwhm2)+c
    return g.ravel()

def two_lorentz(x,c, a0,a1, xo0, xo1, fwhm0, fwhm1):
    g = lorentz(x,a0,xo0,fwhm0)+lorentz(x,a1,xo1,fwhm1)+c
    return g.ravel()

def two_lorentz_asy(x,c, a0,a1, xo0, xo1, fwhm0, fwhm1,asy0,asy1):
    g = asymlorentz(x,a0,xo0,fwhm0,asy0)+asymlorentz(x,a1,xo1,fwhm1,asy1)+c
    return g.ravel()



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

fit_peaks = False
remove_exp = False
plot_details = True
plot_farfield = True

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



peakwl_scat = np.zeros(len(height),dtype=np.object)
peakwl_charge = np.zeros(len(height),dtype=np.object)


mat = sio.loadmat(path + sims[0])
wl = mat['enei'][0]
img = np.zeros((len(wl),len(sims)))


for n,sim in enumerate(sims):
    print(str(sim)+' '+str(n))
    mat = sio.loadmat(path+sim)

    wl = mat['enei'][0]
    sca = np.transpose(mat['sca'])[0]

    img[:,n] = sca

    #p1 = mat['p1']
    #p2 = mat['p2']

    #p = sio.loadmat(path + 'quad_p.mat')

    p =mat['p0']

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


    ##verts = np.vstack( (p1['verts'][0][0],p2['verts'][0][0]))
    ##faces = np.vstack( (p1['faces'][0][0],p2['faces'][0][0]))[:,0:3]
    #verts1 = p1['verts'][0][0]
    #faces1 = p1['faces'][0][0][:,0:3]
    #faces1 = np.array(faces1-1,dtype=np.int)

    #verts2 = p2['verts'][0][0]
    #faces2 = p2['faces'][0][0][:,0:3]
    #faces2 = np.array(faces2-1,dtype=np.int)

    verts = p['verts'][0][0]
    faces = p['faces'][0][0][:,0:3]
    faces = np.array(faces-1,dtype=np.int)

    # x_min = verts[:, 0].min()
    # x_max = verts[:, 0].max()
    # y_min = verts[:, 1].min()
    # y_max = verts[:, 1].max()
    # z_min = verts[:, 2].min()
    # z_max = verts[:, 2].max()

    x_min = -55
    x_max = 55
    y_min = -55
    y_max = 55
    z_min = -5
    z_max = 155

    NA = 0.7
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

    scat /= scat.max()
    indexes_sca = peakutils.indexes(scat, thres=0.0, min_dist=2)

    plt.plot(wl,scat,zorder=0)
    #plt.plot(wl,sca)
    plt.scatter(wl[indexes_sca],scat[indexes_sca],marker='.',c='black',zorder=5)
    for ind in indexes_sca:
        plt.text(wl[ind],scat[ind],str(int(round(wl[ind]))),zorder=10)

    plt.ylabel(r'$I_{scat}\, /\, a.u.$')
    plt.xlabel(r'$\lambda\, /\, nm$')
    plt.tight_layout()
    #plt.savefig(savedir + sim[:-4] + "_scattering.png", dpi=400)
    plt.savefig(savedir + sim[:-4] + "_scattering.eps", dpi=1200)
    #plt.show()
    plt.close()
    peakwl_scat[n] = wl[indexes_sca]

    charge = np.zeros(len(wl))
    for i in range(len(wl)):
        #charge[i] = np.abs((sig2[i]+sig1[i])/2).max()
        charge[i] = np.abs(np.real(sig2[i])).max()
        #charge[i] = np.abs(sig2[i]).max()

    charge /= charge.max()
    indexes_charge = peakutils.indexes(charge, thres=0.1, min_dist=2)
    plt.plot(wl,charge,zorder=0)
    plt.scatter(wl[indexes_charge], charge[indexes_charge],marker='.',c='black',zorder=5)
    for ind in indexes_charge:
        plt.text(wl[ind],charge[ind],str(int(round(wl[ind]))),zorder=10)

    plt.ylabel(r'$\left|\sigma_{2}\right|_{max}\, /\, a.u.$')
    plt.xlabel(r'$\lambda\, /\, nm$')
    plt.tight_layout()
    #plt.savefig(savedir + sim[:-4] + "_maxcharge.png", dpi=400)
    plt.savefig(savedir + sim[:-4] + "_maxcharge.eps", dpi=1200)
    #plt.show()
    plt.close()
    peakwl_charge[n] = wl[indexes_charge]


    if fit_peaks:
        if remove_exp:
            mask = (wl < 500) | (wl > 900)

            x = wl[mask]
            y = sca[mask]

            # fit_fun = lambda x, a,b,c: a*np.exp(b*x)+c
            # p0 = [y.max()*3000,-0.02,0]
            #
            # plt.plot(x,y,linestyle='',marker='.')
            # plt.plot(x,fit_fun(x,p0[0],p0[1],p0[2]),linestyle='',marker='.')
            # plt.show()
            # plt.close()
            # popt, pcov = curve_fit(fit_fun, x, y, p0)
            #
            #
            # plt.plot(x,y,linestyle='',marker='.')
            # plt.plot(x,fit_fun(x,popt[0],popt[1],popt[2]),linestyle='',marker='.')
            # plt.show()
            # plt.close()
            #
            #
            # x = wl
            # y = sca-fit_fun(x,popt[0],popt[1],popt[2])

            fit_fun = lambda x, amp,x0,sigma,c: lorentz(x,amp,x0,sigma)+c
            p0 = [y.max()*2,200,100,0]


            plt.plot(x,y,linestyle='',marker='.')
            plt.plot(x,fit_fun(x,p0[0],p0[1],p0[2],p0[3]),linestyle='',marker='.')
            plt.title('Start Values blue lorentz fit')
            plt.show()
            plt.close()

            popt, pcov = curve_fit(fit_fun, x, y, p0)


            plt.plot(x,y,linestyle='',marker='.')
            plt.plot(x,fit_fun(x,popt[0],popt[1],popt[2],p0[3]),linestyle='',marker='.')
            plt.title('Blue lorentz fit')
            plt.show()
            plt.close()


            x = wl
            y = sca-fit_fun(x,popt[0],popt[1],popt[2],p0[3])
            x = x[wl > 500]
            y = y[wl > 500]

        else:
            x = wl
            y = sca
            x = x[wl > 500]
            y = y[wl > 500]

        skip_three_lorentz = False

        indexes_peaks = peakutils.indexes(y, thres=0.0, min_dist=2)
        print(indexes_peaks)
        if len(indexes_peaks) == 3:
            p0 = [0, y[indexes_peaks[0]], y[indexes_peaks[1]], y[indexes_peaks[2]], x[indexes_peaks[0]], x[indexes_peaks[1]], x[indexes_peaks[2]], 50, 50,50]
        elif len(indexes_peaks) > 1:
            p0 = [0, y[0], y[indexes_peaks[0]], y[indexes_peaks[1]], x[0], x[indexes_peaks[0]], x[indexes_peaks[1]], 50, 50,50]
        else:
            skip_three_lorentz = True

        try:
            if skip_three_lorentz:
                raise RuntimeError('Skip it')

            plt.plot(x, y, linestyle='', marker='.')
            plt.plot(x, three_lorentz(x, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6], p0[7], p0[8], p0[9]),
                     linestyle='', marker='.')
            plt.title('Start values for three lorentz fit')
            plt.show()
            plt.close()

            popt, pcov = curve_fit(three_lorentz, x, y, p0)

            plt.plot(x, y, linestyle='', marker='.')
            plt.plot(x, three_lorentz(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8],
                                      popt[9]))
            plt.title('Three lorentz fit')
            plt.show()
            plt.close()

            peakwl_scat[n] = popt[4:7]
            print("Three Lorentzes at " + str(popt[4:7]))


        except RuntimeError as error:
            print("Two Lorentzes at " +str(indexes_peaks))
            if len(indexes_peaks) > 1:
                #def two_lorentz_asy(x, c, a0, a1, xo0, xo1, fwhm0, fwhm1, asy0, asy1)
                #p0 = [0, y[indexes_peaks[0]]/3, y[indexes_peaks[1]]/2, x[indexes_peaks[0]], x[indexes_peaks[1]], 50, 100,-0.005,-0.005]
                p0 = [0, y[indexes_peaks[0]], y[indexes_peaks[1]], x[indexes_peaks[0]], x[indexes_peaks[1]], 50, 100]

            else:
                #p0 = [0, y[10]/3, y[indexes_peaks[0]]/2, x[10], x[indexes_peaks[0]], 50, 100,-0.005,-0.005]
                p0 = [0, y[indexes_peaks[0]], y[indexes_peaks[0]+10], x[indexes_peaks[0]], x[indexes_peaks[0]+10], 50, 100]

            #bounds = ((-np.inf,0,0,500,500,10,10,-0.01,-0.01), (np.inf,np.inf,np.inf,600,800,1000,1000,0,0))
            bounds = ((-np.inf,0,0,500,500,10,10), (np.inf,np.inf,np.inf,600,800,1000,1000))

            plt.plot(x, y, linestyle='', marker='.')
            plt.plot(x, two_lorentz(x, p0[0], p0[1], p0[2], p0[3], p0[4], p0[5], p0[6]),
                     linestyle='', marker='.')
            plt.show()
            plt.close()

            popt, pcov = curve_fit(two_lorentz, x, y, p0,bounds=bounds,maxfev=16000)
            print(popt)

            plt.plot(x, y, linestyle='', marker='.')
            plt.plot(x, two_lorentz(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]))
            plt.show()
            plt.close()

            peakwl_scat[n] = popt[3:5]
            #print(popt[3:5])


    #verts1[:, 0] += 20
    #verts2[:, 0] -= 20




    if plot_details:
        # import matplotlib
        # matplotlib.use('WXAgg', warn=False, force=True)
        # from matplotlib import pyplot as plt

        #for ind in indexes_sca:
        for ind in indexes_charge:

            # val = np.real(sig2[ind])
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.axis('equal')
            # poly1 = plot_trimesh(verts,faces,val)
            # val1 = np.array(val)  # To be safe - set_array() will expect a numpy array
            # #val1_norm = (val1-val1.min()) / (val1.max() - val1.min())
            # #val1_norm = val1
            # norm = colors.SymLogNorm(vmin=val1.min(), vmax=val1.max(),linthresh=0.1)
            # colormap = cm.seismic(norm(val1))
            # colormap[:, -1] = 0.5
            # poly1.set_facecolor(colormap)
            # poly1.set_edgecolor(None)
            # poly1.set_linewidth(0.0)
            # #ax.add_collection3d(poly1)
            # ax.set_xlim3d(verts[:,0].min(), verts[:,0].max())
            # ax.set_ylim3d(verts[:,1].min(), verts[:,1].max())
            # ax.set_zlim3d(verts[:,2].min(), verts[:,2].max())
            # #ax.autoscale_view()
            # plt.axis('off')
            # plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm.png", dpi=1200)
            # #plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm.pdf", dpi=1200)
            # #plt.show()
            # plt.close()


            val = np.real(sig2[ind])
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(90,90)
            ax.axis('equal')
            poly1 = plot_trimesh(verts,faces,val)
            val1 = np.array(val)  # To be safe - set_array() will expect a numpy array
            val1_norm = (val1-val1.min()) / (val1.max() - val1.min())
            #val1_norm = val1
            norm = colors.SymLogNorm(vmin=val1.min(), vmax=val1.max(),linthresh=0.1)
            colormap = cm.RdBu(norm(val1))
            poly1.set_facecolor(colormap)
            poly1.set_edgecolor('white')
            poly1.set_linewidth(0.0)
            ax.add_collection3d(poly1)
            ax.set_xlim3d(x_min, x_max)
            ax.set_ylim3d(y_min, y_max)
            ax.set_zlim3d(z_min, z_max)
            ax.autoscale_view(tight=True)
            plt.axis('off')
            plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm_top.png", dpi=1200)
            #plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm_top.pdf", dpi=1200)
            #plt.show()
            plt.close()

            val = np.real(sig2[ind])
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(0,90)
            ax.axis('equal')
            poly1 = plot_trimesh(verts,faces,val)
            val1 = np.array(val)  # To be safe - set_array() will expect a numpy array
            val1_norm = (val1-val1.min()) / (val1.max() - val1.min())
            #val1_norm = val1
            norm = colors.SymLogNorm(vmin=val1.min(), vmax=val1.max(),linthresh=0.1)
            colormap = cm.RdBu(norm(val1))
            poly1.set_facecolor(colormap)
            poly1.set_edgecolor('white')
            poly1.set_linewidth(0.0)
            ax.add_collection3d(poly1)
            ax.set_xlim3d(x_min, x_max)
            ax.set_ylim3d(y_min, y_max)
            ax.set_zlim3d(z_min, z_max)
            ax.autoscale_view(tight=True)
            plt.axis('off')
            plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm_side.png", dpi=1200)
            #plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm_side.pdf", dpi=1200)
            #plt.show()
            plt.close()

            val = np.real(sig2[ind])
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(-90,90)
            ax.axis('equal')
            poly1 = plot_trimesh(verts,faces,val)
            val1 = np.array(val)  # To be safe - set_array() will expect a numpy array
            val1_norm = (val1-val1.min()) / (val1.max() - val1.min())
            #val1_norm = val1
            norm = colors.SymLogNorm(vmin=val1.min(), vmax=val1.max(),linthresh=0.1)
            colormap = cm.RdBu(norm(val1))
            poly1.set_facecolor(colormap)
            poly1.set_edgecolor('white')
            poly1.set_linewidth(0.0)
            ax.add_collection3d(poly1)
            ax.set_xlim3d(x_min, x_max)
            ax.set_ylim3d(y_min, y_max)
            ax.set_zlim3d(z_min, z_max)
            ax.autoscale_view(tight=True)
            plt.axis('off')
            plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm_bottom.png", dpi=1200)
            #plt.savefig(savedir + sim[:-4] + "_charge_at_"+ str(int(round(wl[ind]))) +"nm_bottom.pdf", dpi=1200)
            #plt.show()
            plt.close()


    if plot_farfield:
        for ind in indexes_sca:
            fig = plt.figure()

            gs = gridspec.GridSpec(1, 2, width_ratios=[10, 0.5], )

            ax1 = plt.subplot(gs[0], projection="polar", aspect=1.)
            ax2 = plt.subplot(gs[1])

            x = pinfty_pos[:,0]
            y = pinfty_pos[:,1]
            z = pinfty_pos[:,2]

            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)

            val = np.linalg.norm( 0.5 * np.real(np.cross(e[ind], np.conjugate(h[ind]))),axis=1)

            theta2, phi2 = np.meshgrid(np.linspace(0,np.pi/2,32),np.linspace(-np.pi,np.pi,64))
            val2 = interpolate.griddata((theta, phi), val, (theta2, phi2), method='nearest', fill_value=0.0)
            polarplot = ax1.pcolormesh(phi2,theta2, val2,linewidth=0,rasterized=True)
            #ax1.set_rgrids([0.5, 1, 1.5],zorder=10)
            ax1.set_rticks([1])
            #ax1.set_rticks([0.5, 1, 1.5],zorder=10)
            #ax1.set_rlabel_position(90)
            #ax1.grid(True)
            plt.colorbar(polarplot, cax=ax2)
            gs.tight_layout(fig)
            plt.savefig(savedir + sim[:-4] + "_farfield_at_"+ str(int(round(wl[ind]))) +"nm.png", dpi=400)
            #plt.savefig(savedir + sim[:-4] + "_farfield_at_"+ str(int(round(wl[ind]))) +"nm.pgf")
            plt.savefig(savedir + sim[:-4] + "_farfield_at_"+ str(int(round(wl[ind]))) +"nm.pdf", dpi=1200)

            #plt.show()
            plt.close()




if fit_peaks:
    fig, ax1 = plt.subplots()
    for i in range(len(gaps)):
        for peakwl in peakwl_scat[i]:
            ax1.scatter(gaps[i],peakwl,color='C0')

    for i in range(len(gaps)):
        for peakwl in peakwl_charge[i]:
            ax1.scatter(gaps[i],peakwl,color='C1')

    ax1.set_ylabel("Peak Wavelength / nm")
    ax1.set_xlabel("Gap Width / nm")
    plt.tight_layout()
    plt.savefig(savedir + sim[:-4] + "_spectral_shift.png", dpi=400)
    #plt.show()
    plt.close()



    fig, ax1 = plt.subplots()
    for i in range(len(gaps)):
        # for peakwl in peakwl_scat[i]:
        #     ax1.scatter(gaps[i]/diameter,peakwl.max(),color='C0')
        ax1.scatter(gaps[i] / diameter, peakwl_scat[i].max(), color='C0')
    ax1.set_ylabel("Peak Wavelength / nm")
    ax1.set_xlabel("Gap / Diameter")
    plt.tight_layout()
    plt.savefig(savedir + sim[:-4] + "_spectral_shift2.png", dpi=400)
    #plt.show()
    plt.close()





    #img = img.T
    #newfig(0.9)
    fig = plt.figure()
    plt.imshow(img.T, aspect='auto', cmap=plt.get_cmap("viridis"), extent=[wl.min(), wl.max(), gaps.max(), gaps.min()], norm=colors.LogNorm())
    plt.ylabel(r'$gap\, /\, nm$')
    plt.xlabel(r'$\lambda\, /\, nm$')
    plt.tight_layout()
    #plt.savefig(savedir + sim[:-4] + "_image_log.pdf")
    plt.savefig(savedir + sim[:-4] + "_image_log.png", dpi=400)
    # plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
    plt.close()

    #newfig(0.9)
    fig = plt.figure()
    plt.tight_layout()
    plt.imshow(img.T, aspect='auto', cmap=plt.get_cmap("viridis"), extent=[wl.min(), wl.max(), gaps.max(), gaps.min()])
    plt.ylabel(r'$gap\, /\, nm$')

    plt.xlabel(r'$\lambda\, /\, nm$')
    #plt.savefig(savedir + sim[:-4] + "_image.pdf")
    plt.savefig(savedir + sim[:-4] + "_image.png", dpi=400)
    # plt.savefig(savedir + "plots/" + file[-4] + ".pgf")
    plt.close()

    x = []
    y = []
    for i in range(len(gaps)):
        x.append(gaps[i] / diameter)
        y.append(peakwl_scat[i].max())

    x = np.array(x)
    y = np.array(y)

    fit_fun = lambda x, a, tau,c: a * np.exp(-x/tau)+c
    p0 = [y.max(),0.2,0]
    popt, pcov = curve_fit(fit_fun, x, y, p0)
    perr = np.sqrt(np.diag(pcov))

    f = open(savedir + "exponential_fit.txt", 'w')
    f.write("params: ")
    for a in popt:
        f.write(str(a) + ' ')
    f.write("\r\n")
    f.write("errs: ")
    for a in perr:
        f.write(str(a) + ' ')
    f.close()

    fig, ax1 = plt.subplots()

    ax1.plot(x,fit_fun(x,popt[0],popt[1],popt[2]),color="C1",linewidth=0.75,linestyle='--',zorder=0)

    ax1.scatter(x, y, marker='.', s=12, color='C0')
    # (_, caps, _) = ax1.errorbar(x, y, xerr=xerr/diameter,yerr=peaks_err, fmt='.', elinewidth=0.5, markersize=6, capsize=4,color='C0')
    # for cap in caps:
    #     cap.set_markeredgewidth(1)

    #ax1.text(s=r"$y="+str(round(popt[0],2))+"\cdot e^{-x/"+str(round(popt[1],2))+"}+"+str(round(popt[2],2))+"$",xy=(0.15,fit_fun(0.15,popt[0],popt[1],popt[2])))#,xytext=(0.3,660))
    ax1.text(0.17,640,s='$y='+str(int(round(popt[0])))+' \cdot e^{-x/'+str(round(popt[1],2))+'}+'+str(int(round(popt[2])))+'$')
    #ax1.text(0.3,600,s='$y=1 e^{-x/1}+1$')


    ax1.set_ylabel("Peak Wavelength / nm")

    ax1.set_xlabel("Gap / Diameter")

    plt.tight_layout()
    plt.savefig(savedir + "dimer_peaks_simulation.pdf", dpi= 300)
    plt.savefig(savedir + "dimer_peaks_simulation.pgf")
    plt.savefig(savedir + "dimer_peaks_simulation.png", dpi= 400)
    plt.close()


    peaksx = []
    for i in range(len(gaps)):
        peaksx.append(peakwl_scat[i].max())

    peaksx = np.array(peaksx)

    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, img.shape[1])]

    # fig = plt.figure(figsize=(size[0],size[0]*2.5))
    fig, ax = newfig(0.5, 2.5)

    ax.axvline(500, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(600, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(700, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    dist_fac = 0.25
    yticks = []
    labels_waterfall = []
    max_int = 0

    # y_pos = dist/dist.max()
    y_pos = np.linspace(0, 1, len(gaps))

    # img = np.zeros((len(wl),len(sims)))

    mask = (wl >= 450) & (wl <= 800)

    for i in range(img.shape[1]):
        y = img[mask, i]
        y -= y.min()
        y /= img.max()
        # y /= 4

        ax.plot(wl[mask], y + y_pos[i], linewidth=1.0, color=colors[i])

        yticks.append(y_pos[i])

        max_int = np.max([max_int, y.max() + y_pos[i]])

        # ax.scatter(peak_wl,filtered[np.argmax(filtered)]+i*dist_fac,s=20,marker="x",color = colors[i])
        ax.scatter(peaksx[i], y[np.abs(wl[mask] - peaksx[i]).argmin()] + y_pos[i], s=20, marker="x", color=colors[i])
        ax.set_xlim([450, 800])

        labels_waterfall.append(str(round(gaps[i], 1)) + 'nm')

    # print(peaks)
    # print(peaks_err)
    ax.set_ylabel(r'$I_{scat}\, /\, a.u.$')
    ax.set_xlabel(r'$\lambda\, /\, nm$')
    # ax.set_ylim([0, (len(pics)+1)*dist_fac*1.1])
    ax.set_ylim([0, max_int * 1.05])

    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off', labelright='on')
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels_waterfall)

    plt.tight_layout()
    # plt.show()
    plt.savefig(path + "dimer_waterfall_simulation.pdf", dpi=400)
    plt.savefig(path + "dimer_waterfall_simulation.pgf")
    plt.savefig(path + "dimer_waterfall_simulation.png", dpi=400)
    plt.close()