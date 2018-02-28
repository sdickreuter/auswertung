import numpy as np
import pandas as pd
from plotsettings import *
import matplotlib.pyplot as plt



savedir = '/home/sei/Nextcloud/pascal/plots/'



path1 = '/home/sei/Nextcloud/pascal/REM/'

ctc1 = pd.read_csv(path1+'ctc.txt')
radii1 = pd.read_csv(path1+'radii.txt')

files = ctc1.file.tolist()

ed1 = []
for f in files:
    ed1.append(int(f[0:2]))

ed1 = np.array(ed1)
ctc_fwhm1 =  np.array(ctc1.fwhm.tolist())
ctc1 = np.array(ctc1.ctc.tolist())
radii_fwhm1 =  np.array(radii1.fwhm.tolist())
radii1 = np.array(radii1.radius.tolist())
unique_ed1 = np.unique(ed1)


path2 = '/home/sei/Nextcloud/pascal/REMb/'

ctc2 = pd.read_csv(path2+'ctc.txt')
radii2 = pd.read_csv(path2+'radii.txt')

files = ctc2.file.tolist()

ed2 = []
for f in files:
    ed2.append(int(f[0:2]))

ed2 = np.array(ed2)
ctc_fwhm2 =  np.array(ctc2.fwhm.tolist())
ctc2 = np.array(ctc2.ctc.tolist())
radii_fwhm2 =  np.array(radii2.fwhm.tolist())
radii2 = np.array(radii2.radius.tolist())
unique_ed2 = np.unique(ed2)

path3 = '/home/sei/Nextcloud/Fatima2/'

ctc3 = pd.read_csv(path3+'ctc.txt')
ed3 = [30,0,90,30,90]
ed3 = np.array(ed3)
ctc3 = np.array(ctc3.ctc.tolist())

path4 = '/home/sei/Nextcloud/Fatima3/'
radii4 = pd.read_csv(path4+'radii.txt')
ed4 = [30,30,0]
ed4 = np.array(ed4)
radii_fwhm4 =  np.array(radii4.fwhm.tolist())
radii4 = np.array(radii4.radius.tolist())

ed4 = np.append(ed4,90)
radii4 = np.append(radii4,96/2)
radii_fwhm4 = np.append(radii_fwhm4,12/2)

ed4 = np.append(ed4,90)
radii4 = np.append(radii4,97/2)
radii_fwhm4 = np.append(radii_fwhm4,10/2)


fig, ax1 = newfig(0.9)
(_, caps, _) = ax1.errorbar(ed1, ctc1, yerr=ctc_fwhm1 / 2 , fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)
plt.scatter(ed1,ctc1)
(_, caps, _) = ax1.errorbar(ed2, ctc2, yerr=ctc_fwhm2 / 2 , fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C1')
for cap in caps:
    cap.set_markeredgewidth(1)
plt.scatter(ed2,ctc2)
plt.scatter(ed3,ctc3)
plt.xlabel('ED duration / min')
plt.ylabel('center to center distance / nm')
plt.legend(['pascal neu','pascal bachelor','fatima'])
plt.tight_layout()
plt.savefig(savedir+'ctc.png',dpi=1200)
#plt.show()


fig, ax1 = newfig(0.9)
(_, caps, _) = ax1.errorbar(ed1, radii1 * 2, yerr=radii_fwhm1 / 2 * 2, fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C0')
for cap in caps:
    cap.set_markeredgewidth(1)
plt.scatter(ed1,radii1*2,marker='o')
(_, caps, _) = ax1.errorbar(ed2, radii2 * 2, yerr=radii_fwhm2 / 2 * 2, fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C1')
for cap in caps:
    cap.set_markeredgewidth(1)
plt.scatter(ed2,radii2*2,marker='v')
(_, caps, _) = ax1.errorbar(ed4, radii4 * 2, yerr=radii_fwhm4 / 2 * 2, fmt='x', elinewidth=0.5, markersize=6, capsize=4, color='C2')
for cap in caps:
    cap.set_markeredgewidth(1)
plt.scatter(ed4,radii4*2,marker='x')
plt.xlim([-3,ed1.max()*1.05])
plt.ylim([0, 2 * radii2.max() + radii_fwhm2.max() * 1.05])
plt.xlabel(r'$ED\ duration\  /\, min$')
plt.ylabel(r'$mean\ equivalent\ diameter\ /\, nm$')
#plt.legend(['pascal neu','pascal bachelor','fatima'])
#plt.legend(['pascal neu','pascal bachelor','fatima'])
plt.xticks(np.arange(0, 100, 10))
plt.tight_layout()
plt.savefig(savedir+'diameter.png',dpi=1200)
#plt.show()
