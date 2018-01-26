import numpy as np

from plotsettings import *
import matplotlib.pyplot as plt

path = '/home/sei/Promotion/dimerpaper/ACS/plots/'

r_new = lambda x,a,h: (3/4*a*h)**(1/3) * (x )**(2/3)

r = np.linspace(0,50,500)


fig, ax = newfig(0.9)

ax.plot(r,r_new(r,1/3,200)-r)
ax.plot(r,r_new(r,2/5,200)-r)
ax.plot(r,r_new(r,1.0,200)-r)

ax.set_xlabel(r'$r_{p} \, /\, nm$')
ax.set_ylabel(r'$\Delta r \, /\, nm$')
ax.set_aspect('equal')
ax.legend([r'$\alpha = 1/3$',r'$\alpha = 2/5$',r'$\alpha = 1$'])#,title=r'$h_{pillar}$')
# plt.plot(wl, counts)
plt.tight_layout()
plt.savefig(path + "radius_increase.pgf")
plt.savefig(path + "radius_increase.png",dpi=600)
plt.savefig(path + "radius_increase.eps")
plt.close()

print( r_new(40,2/5,200)-40 )


path = '/home/sei/Promotion/dissertation/thesis/plots/radius_increase/'


fig, ax = newfig(0.9)

ax.plot(r,r_new(r,1/3,200)-r)
ax.plot(r,r_new(r,2/5,200)-r,'--')
ax.plot(r,r_new(r,2/3,200)-r)
ax.plot(r,r_new(r,1.0,200)-r)

ax.set_xlabel(r'$r_{S} \, /\, nm$')
ax.set_ylabel(r'$\Delta r \, /\, nm$')
ax.set_aspect('equal')
leg = plt.legend([r'$\alpha = 1/3$',r'$\alpha = 2/5$',r'$\alpha = 2/3$',r'$\alpha = 1$'],fancybox=True)#,title=r'$h_{pillar}$')
leg.get_frame().set_alpha(0.8)
leg.draw_frame(True)
# plt.plot(wl, counts)
plt.tight_layout()
plt.savefig(path + "radius_increase_ger.pgf")
plt.savefig(path + "radius_increase_ger.png",dpi=600)
plt.savefig(path + "radius_increase_ger.eps")
plt.close()





# fig, ax = newfig(0.45,height_mul=2.5)
# #ax.plot(r,r_new(r,100))
# #ax.plot(r,r_new(r,200))
# #ax.plot(r,r_new(r,300))
#
# ax.plot(r,r_new(r,1/3,200))
# ax.plot(r,r_new(r,2/3,200))
# ax.plot(r,r_new(r,1.0,200))
# ax.plot(r,r,'k--',linewidth=0.75)
#
#
# ax.set_xlabel(r'$r_{pillar} \, /\, nm$')
# ax.set_ylabel(r'$r_{sphere} \, /\, nm$')
# ax.set_aspect('equal')
# ax.legend([r'$\alpha = 1/3$',r'$\alpha = 2/3$',r'$\alpha = 1$'])#,title=r'$h_{pillar}$')
# # plt.plot(wl, counts)
# plt.tight_layout()
# plt.savefig(path + "radius_increase.pgf")
# plt.savefig(path + "radius_increase.png",dpi=600)


#plt.show()