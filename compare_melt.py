__author__ = 'sei'

import re
import os
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.4)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#path = '/home/sei/Spektren/rods4/'
#sample1 = 'SiOx2_mono_B_3/'
#sample2 = 'SiOx2_mono_B_melt/'

#samples1 = ['rods4_A0','rods4_A1','rods4_A2','rods4_B0','rods4_B1','rods4_B2']
#samples2 = ['rods4_A0m','rods4_A1m','rods4_A2m','rods4_B0m','rods4_B1m','rods4_B2m']

#path = '/home/sei/Spektren/rods5/'
#samples1 = ['rods5_D0m','rods5_D1m']
#samples2 = ['rods5_D0mm','rods5_D1mm']
#samples3 = ['rods5_D0mmm','rods5_D1mmm']

path = '/home/sei/Spektren/rods6/'
samples1 = ['rods6_A1m']
samples2 = ['rods6_A1m_0pol']
samples3 = ['rods6_A1m_90pol']



for i in range(len(samples1)):


    dir1 = path + samples1[i] + '/specs/'
    dir2 = path + samples2[i] + '/specs/'
    dir3 = path + samples3[i] + '/specs/'
    savedir = path+samples1[i] + '_comparison/'

    try:
        os.mkdir(savedir)
    except:
        pass

    files1 = []
    for file in os.listdir(dir1):
        if re.fullmatch(r"([A-Z]{1}[0-9]{1})(_corr.csv)$", file) is not None:
            files1.append(file)

    files2 = []
    for file in os.listdir(dir2):
        if re.fullmatch(r"([A-Z]{1}[0-9]{1})(_corr.csv)$", file) is not None:
            files2.append(file)

    files3 = []
    for file in os.listdir(dir3):
        if re.fullmatch(r"([A-Z]{1}[0-9]{1})(_corr.csv)$", file) is not None:
            files3.append(file)


    n = len(files1)
    for i in range(n):
        file1 = files1[i]
        file2 = files2[i]
        file3 = files3[i]

        wl, int1 = np.loadtxt(open(dir1+file1,"rb"),delimiter=",",skiprows=6,unpack=True)
        wl, int2 = np.loadtxt(open(dir2+file2,"rb"),delimiter=",",skiprows=6,unpack=True)
        wl, int3 = np.loadtxt(open(dir3+file3,"rb"),delimiter=",",skiprows=6,unpack=True)

        int1, = plt.plot(wl,int1)
        int2, = plt.plot(wl,int2)
        int3, = plt.plot(wl,int3)
        #plt.title('Darkfield Intensity at 532nm')
        plt.ylabel(r'$I_{df}\/[a.u.]$')
        plt.xlabel(r'$\lambda \/ [nm]$')
        #plt.legend([int1, (int1, int2, int3)], ["Normal", "Annealed", "2x Annealed"])
        #plt.legend([int1, int2, int3], ["Annealed", "2x Annealed", "3x Annealed"])
        plt.legend([int1, int2, int3], ["unpol", "0°", "90°"])
        #sns.despine()
        #plt.show()
        plt.savefig(savedir + file1 + ".png", format='png')
        plt.close()
