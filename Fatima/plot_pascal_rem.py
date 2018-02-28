import numpy as np
import pandas as pd
from plotsettings import *
import matplotlib.pyplot as plt

#path = '/home/sei/Nextcloud_Uni/pascal/REMb/'
#path = '/home/sei/Nextcloud_Uni/fatima/'
path = '/home/sei/Nextcloud/pascal/REM/'

ctc = pd.read_csv(path+'ctc.txt')
radii = pd.read_csv(path+'radii.txt')

files = ctc.file.tolist()

ed = []
for f in files:
    ed.append(int(f[0:2]))

ed = np.array(ed)
ctc_fwhm =  np.array(ctc.fwhm.tolist())
ctc = np.array(ctc.ctc.tolist())
radii_fwhm =  np.array(radii.fwhm.tolist())
radii = np.array(radii.radius.tolist())
unique_ed = np.unique(ed)

# r_mean = []
# r_err_pos = []
# r_err_neg = []
# d_mean = []
# d_err_pos = []
# d_err_neg = []
#
# for a in unique_ed:
#     mask = (ed == a)
#     r_mean.append(np.mean(radii[mask]))
#     r_err_pos.append(np.max(radii[mask]) + np.mean(fwhm[mask]/ 2) )
#     r_err_neg.append(np.min(radii[mask]) - np.mean(fwhm[mask]/ 2) )
#
#     d_mean.append(np.mean(radii[mask]*2))
#     d_err_pos.append(np.max(radii[mask]*2) + np.mean(fwhm[mask]/ 2 * 2) )
#     d_err_neg.append(np.min(radii[mask]*2) - np.mean(fwhm[mask]/ 2 * 2) )
#
# r_mean = np.array(r_mean)
# r_err_pos = np.array(r_err_pos)
# r_err_neg = np.array(r_err_neg)
# d_mean = np.array(d_mean)
# d_err_pos = np.array(d_err_pos)
# d_err_neg = np.array(d_err_neg)

fig, ax1 = newfig(0.9)
(_, caps, _) = ax1.errorbar(ed, ctc, yerr=ctc_fwhm / 2 , fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)
plt.scatter(ed,ctc)
plt.xlabel('ED duration / min')
plt.ylabel('center to center distance / nm')
plt.tight_layout()
plt.savefig(path+'ctc.png',dpi=1200)
#plt.show()

fig, ax1 = newfig(0.9)
# (_, caps, _) = ax1.errorbar(unique_ed, r_mean, yerr=(r_err_pos, r_err_neg), fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C0')
# for cap in caps:
#     cap.set_markeredgewidth(1)
(_, caps, _) = ax1.errorbar(ed, radii, yerr=radii_fwhm / 2, fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)
plt.scatter(ed,radii)
plt.xlabel('ED duration / min')
plt.ylabel('mean equivalent radius / nm')
plt.tight_layout()
plt.savefig(path+'radii.png',dpi=1200)
#plt.show()

fig, ax1 = newfig(0.9)
# (_, caps, _) = ax1.errorbar(unique_ed, d_mean, yerr=(d_err_pos, d_err_neg), fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C0')
# for cap in caps:
#     cap.set_markeredgewidth(1)
(_, caps, _) = ax1.errorbar(ed, radii * 2, yerr=radii_fwhm / 2 * 2, fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)
plt.scatter(ed,radii*2)
plt.xlim([0,ed.max()*1.05])
plt.ylim([0, 2 * radii.max() + radii_fwhm.max() * 1.05])
plt.xlabel('ED duration / min')
plt.ylabel('mean equivalent diameter / nm')
plt.tight_layout()
plt.savefig(path+'diameter.png',dpi=1200)
#plt.show()