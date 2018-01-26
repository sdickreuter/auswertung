import numpy as np
import pandas as pd
from plotsettings import *
import matplotlib.pyplot as plt

path = '/home/sei/Nextcloud_Uni/pascal/REM/'


ctc = pd.read_csv(path+'ctc.txt')
radii = pd.read_csv(path+'radii.txt')

files = ctc.file.tolist()

ed = []
for f in files:
    ed.append(int(f[0:2]))

ed = np.array(ed)
ctc = np.array(ctc.ctc.tolist())
radii = np.array(radii.radius.tolist())

newfig(0.9)
plt.scatter(ed,ctc)
plt.xlabel('ED duration / min')
plt.ylabel('center to center distance / nm')
plt.tight_layout()
plt.savefig(path+'ctc.png',dpi=1200)
#plt.show()

newfig(0.9)
plt.scatter(ed,radii)
plt.xlabel('ED duration / min')
plt.ylabel('mean equivalent radius / nm')
plt.tight_layout()
plt.savefig(path+'radii.png',dpi=1200)
#plt.show()

newfig(0.9)
plt.scatter(ed,radii*2)
plt.xlabel('ED duration / min')
plt.ylabel('mean equivalent diameter / nm')
plt.tight_layout()
plt.savefig(path+'diameter.png',dpi=1200)
#plt.show()