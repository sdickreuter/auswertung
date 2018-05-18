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
import matplotlib.colors as colors
import matplotlib.cm as cm
#import seaborn as sns
from scipy.spatial import Delaunay
from scipy import signal
from scipy import interpolate
from matplotlib import gridspec
from adjustText import adjust_text

#colormap = plt.get_cmap('tab20c')
#mpl.rcParams['axes.color_cycle'] = [colormap(k) for k in np.linspace(0, 1, 5)]


#file1 = '/home/sei/MNPBEM/single_horzpol_nosub/single_horzpol_nosub_d0nm_r45nm_theta0.mat'
#file2 = '/home/sei/MNPBEM/dimer_horzpol/dimer_horzpol_r45nm_d2nm_theta0.mat'

file1 = '/home/sei/MNPBEM/dimer_1nmOx/dimer_r45nm_d2nm_theta45.mat'
file2 = '/home/sei/MNPBEM/dimer_90nmOx/dimer_r45nm_d2nm_theta45_90nmOx.mat'
file3 = '/home/sei/MNPBEM/dimer_onlyox/dimer_onlyox_r45nm_d2nm_theta45_1nmOx.mat'


savename = 'oxide_vergleich'#"sca_vergleich"
legend1 = ['$1 \;$nm Oxid $\cdot10$','$90 \;$nm Oxid','nur Oxid $\cdot5$']#['Einzel $\cdot10$','Dimer']
legend2 = ['$1 \;$nm Oxid','$90 \;$nm Oxid','nur Oxid']#['Einzel $\cdot10$','Dimer']
savedir = './plots/'
annotate = False

try:
    os.mkdir(savedir)
except:
    pass


mat = sio.loadmat(file1)
wl = mat['enei'][0]


mat = sio.loadmat(file1)
sca1 = np.transpose(mat['sca'])[0]

mat = sio.loadmat(file2)
sca2 = np.transpose(mat['sca'])[0]

mat = sio.loadmat(file3)
sca3 = np.transpose(mat['sca'])[0]


fig, ax = newfig(0.9)

ax.plot(wl, sca1*10, zorder=0,linestyle='--')
ax.plot(wl, sca2, zorder=1,linestyle='-')
ax.plot(wl, sca3*5, zorder=1,linestyle=':')

plt.ylabel(r'$I_{scat}\, /\, a.u.$')
plt.xlabel(r'$\lambda\, /\, nm$')
plt.legend(legend1)
plt.tight_layout()
# plt.savefig(savedir + sim[:-4] + "_scattering.png", dpi=400)
plt.savefig(savedir + savename+"_sca.eps", dpi=1200)
plt.savefig(savedir + savename+"_sca.pgf")
# plt.show()
plt.close()

mat = sio.loadmat(file1)
sig1 = np.zeros(len(wl), dtype=np.object)

for i in range(len(wl)):
    sig = mat['sigs']
    sig1[i] = sig[0, i]['sig2'][0][0].T[0]

charge1 = np.zeros(len(wl))
for i in range(len(wl)):
    charge1[i] = np.abs(np.real(sig1[i])).max()

mat = sio.loadmat(file2)
sig2 = np.zeros(len(wl), dtype=np.object)

for i in range(len(wl)):
    sig = mat['sigs']
    sig2[i] = sig[0, i]['sig2'][0][0].T[0]

charge2 = np.zeros(len(wl))
for i in range(len(wl)):
    charge2[i] = np.abs(np.real(sig2[i])).max()

mat = sio.loadmat(file3)
sig3 = np.zeros(len(wl), dtype=np.object)

for i in range(len(wl)):
    sig = mat['sigs']
    sig3[i] = sig[0, i]['sig2'][0][0].T[0]

charge3 = np.zeros(len(wl))
for i in range(len(wl)):
    charge3[i] = np.abs(np.real(sig3[i])).max()


fig, ax = newfig(0.9)


ax.plot(wl, charge1, zorder=0,linestyle='--')
ax.plot(wl, charge2, zorder=1,linestyle='-')
ax.plot(wl, charge3, zorder=2,linestyle=':')



plt.ylabel(r'$\left|\sigma_{2}\right|_{max}\, /\, a.u.$')
plt.xlabel(r'$\lambda\, /\, nm$')
plt.legend(legend2)
plt.tight_layout()
# plt.savefig(savedir + sim[:-4] + "_scattering.png", dpi=400)
plt.savefig(savedir + savename+"_sig.eps", dpi=1200)
plt.savefig(savedir + savename+"_sig.pgf")
# plt.show()
plt.close()
