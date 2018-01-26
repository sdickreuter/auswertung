import numpy as np

file = "/home/sei/matlab/MNPBEM17/Material/@epstable/siliconIR_um.dat"

data = np.loadtxt(file,comments="%")

print(data[:,0])

data[:,0] = 1.2398/data[:,0]

print(data[:,0])

np.savetxt(file[:-7]+".dat",data)