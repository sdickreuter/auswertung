
import numpy as np
from plotsettings import *
import os
import re
import PIL
import matplotlib.gridspec as gridspec
#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

#sample = "p41m"
#arrays = ["dif0", "dif1", "dif2", "dif3", "dif4", "dif5", "dif6"]
#arrays = ["dif5"]

#sample = "p52m"
#arrays = ["dif0", "dif1", "dif2", "dif3", "dif5", "dif6"]
#arrays = ["dif5"]

#sample = "p45m"
#arrays = ["did4", "did5", "did6"]
#arrays = ["did5"]

#sample = "2C1"
#arrays = ["2C1_75hept_B2"]

sample = "8"
arrays = ["C3","C4","D2","D3"]


letters = [chr(c) for c in range(65, 91)]

for array in arrays:

    #path = '/home/sei/Auswertung/'+sample+'/'+array+'/'
    #path = '/home/sei/REM/' + array+'/'
    path = '/home/sei/REM/' + sample + '/'

    # try:
    #     nmpx = np.loadtxt(path + "nmppx")
    # except:
    #     raise RuntimeError("nmppx not found!")
    nmpx = 2.63271  # nm/px

    print(nmpx)


    #savedir = 'plots/'
    #savedir = 'denoised/'
    savedir = array+'_plots/'

    fname = path + "sem_overview.jpg"

    try:
        os.mkdir(path+"overview/")
    except:
        pass

    # files = []
    # for file in os.listdir(path+savedir):
    #     if re.fullmatch(r"([A-Z]{1}[1-9]{1})(.png)$", file) is not None:
    #         files.append(file)
    #
    # files.sort()
    # print(files)

    #n = int(np.sqrt(len(files)))
    nx = 5
    ny = 5

    #fig, axs = plt.subplots(n,n, figsize=figsize(1.0))
    #fig.subplots_adjust(hspace = .001, wspace=.001)

    size = figsize(1.0)
    #fig = plt.figure(figsize=(size[0], size[0]))
    fig = plt.figure(figsize=(size[0],size[0]*1.0))
    #fig.suptitle(sample+' '+array)
    gs1 = gridspec.GridSpec(nx, ny,wspace=0.01, hspace=0.01)
    #indices = np.arange(0,len(files),1)


    c=0
    for ix in range(nx):
        for iy in range(ny):
            i = (ix)+(iy*ny)
            id = (letters[iy] + "{0:d}".format(ix+1))
            try:
                #print(path + savedir + id+'_denoised.png')
                img = np.asarray(PIL.Image.open(path + savedir + id+'.png'))
                #img = np.asarray(PIL.Image.open(path + savedir + id + '_denoised.jpg'))
                #img = img[40:(160 - 40), 40:(160 - 40)]
                #img = img[0:(484-0),60:(712-60)]
                skip = False
            except:
                img = np.zeros((100,100))
                skip = True

            img = 255 - img
            #img = np.flipud(img)
            ax = plt.Subplot(fig,gs1[c])
            #ax.text(1, img.shape[1] - 1, id, color="white", fontsize=8)
            if not skip:
                ax.imshow(img, cmap="gray_r")
                #ax.text(2, +10, id, color="white", fontsize=8)
                #ax.text(10, +65, id, color="white", fontsize=8)
                #ax.text(0, -20, id, color="black", fontsize=8)
                ax.text(2, +17, id, color="white", fontsize=8)
                scalebar = AnchoredSizeBar(ax.transData,
                                           100/nmpx, '', 'lower right',
                                           pad=0.2,
                                           color='white',
                                           frameon=False,
                                           size_vertical=1,label_top=True)
                ax.add_artist(scalebar)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
            fig.add_subplot(ax)
            # ax.set_title(file[:-4])
            c += 1

    # c=0
    # for ix in range(nx):
    #     for iy in range(ny):
    #         i = (ix)+(iy*ny)
    #         print(files[i])
    #         file = files[i]
    #         img = np.asarray(PIL.Image.open(path + savedir + file))
    #         img = 255 - img
    #         ax = plt.subplot(gs1[c])
    #         ax.imshow(img)
    #         ax.text(1, img.shape[1] - 1, file[:-4], color="white", fontsize=8)
    #         ax.set_axis_off()
    #         # ax.set_title(file[:-4])
    #         c += 1

    # for i,file in enumerate(files):
    #     print(i)
    #     img = np.asarray(PIL.Image.open(path+savedir+file))
    #     img = 255-img
    #     ax = plt.subplot(gs1[i])
    #     ax.imshow(img)
    #     ax.text(1,img.shape[1]-1,file[:-4],color="white",fontsize=8)
    #     ax.set_axis_off()
    #     #ax.set_title(file[:-4])

    #gs1.update(wspace=0.01, hspace=0.0) # set the spacing between axes.
    #gs1.tight_layout(fig,pad=1.0,h_pad=-0.5, w_pad=0.2)
    gs1.tight_layout(fig, pad=0.1, h_pad=0.1, w_pad=0.1)

    #plt.show()
    # plt.savefig(path+"overview/" + sample+'_'+array+ "_sem_overview.pdf", dpi= 300)
    # plt.savefig(path+"overview/" + sample+'_'+array+"_sem_overview.pgf")
    # plt.savefig(path+"overview/" + sample+'_'+array+"_sem_overview.eps", dpi = 1200)
    plt.savefig(path+"overview/" + array+ "_sem_overview.pdf", dpi= 1200)
    plt.savefig(path+"overview/" + array+"_sem_overview.pgf", dpi= 2400)
    plt.savefig(path+"overview/" + array+"_sem_overview.eps", dpi = 1200)
    plt.close()