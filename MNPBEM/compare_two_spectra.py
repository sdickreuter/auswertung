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

savename = 'oxide_vergleich'#"sca_vergleich"
legend1 = ['$1 \;$nm Oxid $\cdot10$','$90 \;$nm Oxid']#['Einzel $\cdot10$','Dimer']
legend2 = ['$1 \;$nm Oxid','$90 \;$nm Oxid']#['Einzel $\cdot10$','Dimer']
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

fig, ax = newfig(0.9)

ax.plot(wl, sca1*10, zorder=0,linestyle='--')
ax.plot(wl, sca2, zorder=1,linestyle='-')

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


indexes_charge1 = peakutils.indexes(charge1, thres=0.1, min_dist=2)
indexes_charge2 = peakutils.indexes(charge2, thres=0.1, min_dist=2)

print(indexes_charge2)


fig, ax = newfig(0.9)

if annotate:
    arrow_prop = dict(facecolor='black', shrink=0.05,width=0.2,headwidth=2.0,headlength=2.0)

ax.plot(wl, charge1, zorder=0,linestyle='--')
if annotate:
    ax.annotate('E1', xy=(wl[indexes_charge1[0]], charge1[indexes_charge1[0]]*10), xytext=(420, 6),arrowprops=arrow_prop)

ax.plot(wl, charge2, zorder=1,linestyle='-')
if annotate:
    ax.annotate('D2', xy=(wl[indexes_charge2[0]], charge2[indexes_charge2[0]]), xytext=(450, 9),arrowprops=arrow_prop)
    ax.annotate('D1', xy=(wl[indexes_charge2[1]], charge2[indexes_charge2[1]]), xytext=(550, 12),arrowprops=arrow_prop)



plt.ylabel(r'$\left|\sigma_{2}\right|_{max}\, /\, a.u.$')
plt.xlabel(r'$\lambda\, /\, nm$')
plt.legend(legend2)
plt.tight_layout()
# plt.savefig(savedir + sim[:-4] + "_scattering.png", dpi=400)
plt.savefig(savedir + savename+"_sig.eps", dpi=1200)
plt.savefig(savedir + savename+"_sig.pgf")
# plt.show()
plt.close()

print("E1: " + str(wl[indexes_charge1[0]]))
print("D1: " + str(wl[indexes_charge2[1]]))
print("D2: " + str(wl[indexes_charge2[0]]))
