import os
import re
import sys

import numpy as np

from plotsettings import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#https://csullender.com/scholar/

Year = np.array([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017],dtype=np.int)

#Results = np.array([129,147,200,272,511,1030,1680,2620,3840,5130,7060,9610,12600,15600,19000,22000,24100,22900],dtype=np.int)
Results = np.array([25,46,42,129,195,350,668,1150,2250,2400,3500,4550,6020,7240,8300,9090,9670,10400],dtype=np.int) # "Plasmonics"





fig, ax = newfig(0.9)
ax.plot(Year, Results)

ax.set_xticks([2000,2005,2010,2015])
#ax.set_yticks([5000,10000,15000,20000,25000])
#ax.set_yticklabels(['5k','10k','15k','20k','25k'])
ax.set_yticks([2000,4000,6000,8000,10000])
ax.set_yticklabels(['2k','4k','6k','8k','10'])

#formatter = mticker.ScalarFormatter(useMathText=True)
#formatter.set_powerlimits((0,0))
#ax.yaxis.set_major_formatter(formatter)
#gca().get_yaxis().get_major_formatter().set_powerlimits((0, 0))

plt.ylabel(r'Suchanfragen')
plt.xlabel(r'Jahr')
#plt.title(r'Suchanfragen für \glqq Plasmonics\grqq{}')
plt.tight_layout()
# plt.savefig(savedir + sim[:-4] + "_scattering.png", dpi=400)
plt.savefig("googletrend.eps", dpi=1200)
plt.savefig("googletrend.pgf")
# plt.show()
plt.close()