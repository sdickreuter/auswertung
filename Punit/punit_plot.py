import os

import numpy as np
from plotsettings import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

path = '/home/sei/Punit_Report/'

raman=pd.read_csv('/home/sei/Punit_Report/Raman/' + 'raman.csv')

print(raman)


rem = pd.read_csv('/home/sei/Punit_Report/SEM/' + 'coverage.csv')

print(rem)


#df = pd.read_csv('/home/sei/Spektren/fatima/' + 'values.csv')
#print(df)


mean = np.array([raman['mean1075'],raman['mean1585']]).T
err = np.array([raman['err1075'],raman['err1585']]).T

area = np.array(rem['area_mean']).T
coverage = np.array(rem['coverage']).T

label = np.array(raman['label']).T

sort = np.argsort(area)
mean = mean[sort,:]
err = err[sort,:]
area = area[sort]
coverage = coverage[sort]
label = label[sort]

diameter = 2*np.sqrt(area/np.pi)

ticklabels = label
for i in range(len(ticklabels)):
    ticklabels[i] = r'' + ticklabels[i] + '\n'+'$\diameter='+str(np.round(diameter[i]))+'nm$'


print(area)

for i in range(mean.shape[1]):
    plt.bar(np.arange(0,mean.shape[0],1)*mean.shape[1]+(i+1),mean[:,i].ravel(),yerr=err[:,i].ravel())
plt.xticks((np.arange(0,mean.shape[0],1)*mean.shape[1]+(mean.shape[1]+1)/2), ticklabels)
plt.ylabel(r'$I_{\nu}\, /\, counts$')
plt.legend(("1085","1590"))
plt.savefig(path+"raman.pdf")
plt.close()

mean_corr = np.zeros(mean.shape)
err_corr = np.zeros(mean.shape)
for i in range(mean.shape[1]):
    mean_corr[:,i] = mean[:,i] / coverage
    err_corr[:,i] = err[:,i] / coverage

f = open(path + "raman_norm.csv", 'w')
f.write("id,mean1085,err1085,mean1590,err1590" + "\r\n")
for i in range(mean.shape[0]):
    f.write(label[i] + ',' + str(mean_corr[i,0]) + "," + str(err_corr[i,0]) + ',' + str(mean_corr[i,1]) + "," + str(err_corr[i,1]) + "\r\n")
f.close()

for i in range(mean.shape[1]):
    plt.bar(np.arange(0,mean.shape[0],1)*mean.shape[1]+(i+1),mean_corr[:,i].ravel(),yerr=err_corr[:,i].ravel())
plt.xticks((np.arange(0,mean.shape[0],1)*mean.shape[1]+(mean.shape[1]+1)/2), ticklabels)
plt.ylabel(r'$I_{\nu}\, /\, counts$')
plt.legend(("1085","1590"))
plt.savefig(path+"normalized_raman.pdf")
plt.close()

# mean_corr = np.zeros(mean.shape)
# err_corr = np.zeros(mean.shape)
# for i in range(mean.shape[1]):
#     mean_corr[:,i] = mean[:,i] / rem['n']
#     err_corr[:,i] = err[:,i] / rem['n']

# for i in range(mean.shape[1]):
#     plt.bar(np.arange(0,mean.shape[0],1)*mean.shape[1]+(i+1),mean_corr[:,i].ravel(),yerr=err_corr[:,i].ravel())
# plt.xticks((np.arange(0,mean.shape[0],1)*mean.shape[1]+(mean.shape[1]+1)/2), label)
# plt.ylabel(r'$I_{\nu}\, /\, counts$')
# plt.legend(("1085","1590"))
# plt.savefig(path+"particle_raman.pdf")
# plt.close()

# mean_corr = np.zeros(mean.shape)
# err_corr = np.zeros(mean.shape)
# for i in range(mean.shape[1]):
#     mean_corr[:,i] = mean[:,i] / (rem['n'] * rem['area_mean'])
#     err_corr[:,i] = err[:,i]  / (rem['n'] * rem['area_mean'])

# for i in range(mean.shape[1]):
#     plt.bar(np.arange(0,mean.shape[0],1)*mean.shape[1]+(i+1),mean_corr[:,i].ravel(),yerr=err_corr[:,i].ravel())
# plt.xticks((np.arange(0,mean.shape[0],1)*mean.shape[1]+(mean.shape[1]+1)/2), label)
# plt.ylabel(r'$I_{\nu}\, /\, counts$')
# plt.legend(("1085","1590"))
# plt.savefig(path+"particlearea_raman.pdf")
# plt.close()

# mean_corr = np.zeros(mean.shape)
# err_corr = np.zeros(mean.shape)


# for i in range(mean.shape[1]):
#     plt.bar(np.arange(0,mean.shape[0],1)*mean.shape[1]+(i+1),mean_corr[:,i].ravel(),yerr=err_corr[:,i].ravel())
# plt.xticks((np.arange(0,mean.shape[0],1)*mean.shape[1]+(mean.shape[1]+1)/2), label)
# plt.ylabel(r'$I_{\nu}\, /\, counts$')
# plt.legend(("1085","1590"))
# plt.savefig(path+"particlearea_calc_raman.pdf")
# plt.close()

# mean_spec = np.array([df['mean630'],df['mean637'],df['mean640']]).T
# err_spec = np.array([df['err630'],df['err637'],df['err640']]).T
#
# values = np.zeros((mean_spec.shape[0],mean_spec.shape[1]-1))
#
# values[:,0] = mean_spec[:,0]*mean_spec[:,1]
# values[:,1] = mean_spec[:,0]*mean_spec[:,2]
#
# values_err = np.zeros((mean_spec.shape[0],mean_spec.shape[1]-1))
#
# values_err[:,0] = err_spec[:,0]*err_spec[:,1]
# values_err[:,1] = err_spec[:,0]*err_spec[:,2]
#
#
#
# for i in range(values.shape[1]):
#     plt.bar(np.arange(0,values.shape[0],1)*values.shape[1]+(i+1),values[:,i].ravel(),yerr=values_err[:,i].ravel())
# plt.xticks((np.arange(0,values.shape[0],1)*values.shape[1]+(values.shape[1]+1)/2), label)
# plt.ylabel(r'$I^{df}_{laser} \cdot I^{df}_{stokes}\, /\, a.u.$')
# plt.legend(("1085","1590"))
# plt.savefig(path+"df_factor.pdf")
# plt.close()
#
#
# for i in range(values.shape[1]):
#     plt.bar(np.arange(0,values.shape[0],1)*values.shape[1]+(i+1),values[:,i].ravel()/ rem['coverage'],yerr=values_err[:,i].ravel()/ rem['coverage'])
# plt.xticks((np.arange(0,values.shape[0],1)*values.shape[1]+(values.shape[1]+1)/2), label)
# plt.ylabel(r'$I^{df}_{laser} \cdot I^{df}_{stokes}\, /\, a.u.$')
# plt.legend(("1085","1590"))
# plt.savefig(path+"df_factor_normalized.pdf")
# plt.close()


#label = np.array(label)
#print(label)


# for i in range(values.shape[1]):
#     plt.scatter(values[:,i].ravel(),mean[:,i].ravel())
#
# for i in range(values.shape[0]):
#     plt.text(values[i,0], mean[i,0]+np.max(mean)*0.02, label[i], fontsize=8)
#
# plt.legend(("1085","1590"))
# plt.xlim((0, values.ravel().max()*1.1))
# plt.ylabel(r'$I_{\nu}\, /\, counts$')
# plt.xlabel(r'$I^{df}_{laser} \cdot I^{df}_{stokes}\, /\, a.u.$')
# plt.savefig(path+"correlation.pdf")
# plt.close()

