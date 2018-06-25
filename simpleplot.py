import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.4)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.signal import savgol_filter
import glob
import pandas as pd
import os

#path = '/home/sei/Spektren/lamptest'
#path = '/home/sei/Spektren/objtest'
#path = '/home/sei/Spektren/pellicletest'
#path = '/home/sei/data/Pascal'
#path = '/home/sei/Spektren/zeisstest'
path = '/home/sei/Spektren/Scan41A111ldb'



files = glob.glob(path+'/*.csv')
#files.sort(key=os.path.getmtime)
files.sort()

for file in files:
    wl, counts = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=16, unpack=True)

    plt.figure(figsize=(8, 6))
    plt.plot(wl, counts)
    plt.ylabel(r'$I_{df} [a.u.]$')
    plt.xlabel(r'$\lambda [nm]$')
    # plt.plot(wl, counts)
    #plt.savefig(file[:-4] + ".png")
    plt.show()
    plt.close()


y = np.zeros((len(files),len(counts)))
x = np.zeros((len(files),len(counts)))
labels = []


for i in range(len(files)):
    wl, counts = np.loadtxt(open(files[i], "rb"), delimiter=",", skiprows=16, unpack=True)
    counts = counts - np.min(counts)
    counts = counts/np.max(counts)
    y[i,:] = counts
    x[i,:] = wl

    # with open(files[i]) as f:
    #     lines = []
    #     i = 0
    #     for line in f:
    #         lines.append(line)
    #         i += 1
    #         if i > 2:
    #             break
    #     labels.append(lines[0][:-2]+' '+lines[1][:-8])
    label = files[i].split('/')[-1][:-4]
    label = label.replace('_',' ')
    labels.append(label)


print(labels)

x = np.flipud(x)
y = np.flipud(y)
labels.reverse()

plt.figure(dpi=600)
plt.ylabel(r'$I_{df} [a.u.]$')
plt.xlabel(r'$\lambda [nm]$')
for i in range(len(files)):
    plt.plot(x[i,:], y[i,:]+i/2)
    plt.text(315,0.1+i/2,labels[i],fontsize=16)
plt.savefig(path+"/Overview.png")
plt.close()

plt.figure(dpi=600)
plt.ylabel(r'$I_{df} [a.u.]$')
plt.xlabel(r'$\lambda [nm]$')
for i in range(len(files)):
    plt.plot(x[i,:], y[i,:],label=labels[i])
    #plt.text(315,0.1+i/2,labels[i],fontsize=16)
plt.legend(loc='upper left')
plt.savefig(path+"/Overview_legend.png")
plt.close()