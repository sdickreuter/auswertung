import numpy as np

from plotsettings import *

import os
import re
from scipy.signal import savgol_filter

#path = '/home/sei/Spektren/lamptest'
#path = '/home/sei/Spektren/p56m_dia6/'
#path = '/home/sei/Spektren/p56m_dia6_2/'
#path = '/home/sei/Spektren/p57m_did6/'
#path = '/home/sei/Spektren/p57m_did6_2/'

#path = '/home/sei/Spektren/p57m_did6/'
#path = '/home/sei/Spektren/p57m_did5/'
#path = '/home/sei/Spektren/p57m_did4/'
#path = '/home/sei/Spektren/p45m_did5/'
#path = '/home/sei/Spektren/p41m_dif0/'
#path = '/home/sei/Spektren/p41m_dif6/'
#path = '/home/sei/Spektren/p41m_dif3/'

#path = '/home/sei/Spektren/A1_oldlense/'
path = '/home/sei/Spektren/A1_newlense_newsetup_100um/'


maxwl = 1010
minwl = 450

wl, lamp = np.loadtxt(open(path + "lamp.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, dark = np.loadtxt(open(path + "dark.csv", "rb"), delimiter=",", skiprows=16, unpack=True)
wl, bg = np.loadtxt(open(path + "background.csv", "rb"), delimiter=",", skiprows=16, unpack=True)

mask = (wl >= minwl) & (wl <= maxwl)

#plt.plot(wl, lamp - dark)
#plt.savefig(path + "lamp.png")
# plt.close()
# plt.plot(wl[mask], bg[mask])
# plt.savefig(path + "plots/bg.png")
# plt.close()
# plt.plot(wl[mask], dark[mask])
# plt.savefig(path + "plots/dark.png")
# plt.close()

files = []
z = []
for file in os.listdir(path):
    #if re.fullmatch(r"([0-9]{5})(.csv)$", file) is not None:
    #if re.fullmatch(r"([A-Z]{1}[1-9]{1})(.csv)$", file) is not None:
    #if re.fullmatch(r"[/w](.csv)$", file) is not None:
    #if re.fullmatch(r"((z)[0-9]{3})(um.csv)$", file) is not None:
    if re.fullmatch(r"([0-9]{1,2})(.csv)$", file) is not None:
        files.append(file)
        m = re.match(r"([0-9]{1,2})", file)
        z.append(int(m.group(0)))

#files = ["0.csv","+1.csv","-1.csv","+300x.csv","-300x.csv","+300y.csv","-300y.csv"]

#files = glob.glob(path+'/*.csv')
#files.sort(key=os.path.getmtime)

# try:
#     os.mkdir(path + "plots/")
# except:
#     pass
z = np.array(z)
files = np.array(files)
sorted_ind = np.argsort(z)
z = z[sorted_ind]
files = files[sorted_ind]

#files = np.flip(files,0)
print(z)
print(files)


# for file in files:
#     wl, counts = np.loadtxt(open(path+file, "rb"), delimiter=",", skiprows=16, unpack=True)
#     wl = wl[mask]
#     counts = (counts-bg)/(lamp-dark)
#     #counts = (counts-dark)/(lamp-dark)
#     counts = counts[mask]
#
#     fig = newfig(0.9)
#     plt.plot(wl, counts)
#     plt.xlim((minwl,maxwl))
#     plt.ylabel(r'$I_{df} [a.u.]$')
#     plt.xlabel(r'$\lambda [nm]$')
#     plt.tight_layout()
#     # plt.plot(wl, counts)
#     plt.savefig(path+file[:-4] + ".png",dpi=300)
#     plt.close()



img = np.zeros((lamp[mask].shape[0],len(files)))

for i in range(len(files)):
    file = files[i]
    wl, counts = np.loadtxt(open(path+file, "rb"), delimiter=",", skiprows=16, unpack=True)
    wl = wl[mask]
    counts = (counts-bg)/(lamp-dark)
    counts = savgol_filter(counts,51,0)
    #counts = (counts-dark)/(lamp-dark)
    counts = counts[mask]
    img[:,i] = counts
    #plt.plot(wl, counts,color=colors[i],linewidth=0.6)

fig = newfig(0.9)
cmap = plt.get_cmap('plasma')#sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
colors = cmap(np.linspace(0.1, 1, len(files)))

for i in range(img.shape[1]):
    #plt.plot(wl, img[:,i]+i*0.001,linewidth=1,color=colors[i])
    plt.plot(wl, img[:, i] + i * 0.0003, linewidth=1, color=colors[i],zorder=img.shape[1]-i)

#plt.plot(wl, meanspec[mask], color = "black", linewidth=1)
plt.xlim((minwl, maxwl))
plt.ylabel(r'$I_{df} [a.u.]$')
plt.xlabel(r'$\lambda [nm]$')
#plt.legend(files)
plt.tight_layout()
plt.savefig(path +'Overview' + ".png",dpi=300)
plt.close()

plt.imshow(img.T,extent=[wl.min(),wl.max(),0,len(files)],aspect=10,cmap='plasma')
plt.xlabel(r'$\lambda [nm]$')
plt.ylabel(r'$z [\mu m]$')
#plt.legend(files)
plt.tight_layout()
plt.savefig(path +'img' + ".png",dpi=300)
plt.close()

plt.imshow(np.log(img.T),extent=[wl.min(),wl.max(),0,len(files)],aspect=10,cmap='plasma')
plt.xlabel(r'$\lambda [nm]$')
plt.ylabel(r'$z [\mu m]$')
#plt.legend(files)
plt.tight_layout()
plt.savefig(path +'img_log' + ".png",dpi=300)
plt.close()