__author__ = 'sei'

import os
#import matplotlib
#matplotlib.use("qt4")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from skimage import exposure
from skimage import measure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.io import imsave
from skimage.measure import regionprops
from skimage.morphology import reconstruction
from skimage.morphology import remove_small_objects
from skimage.morphology import watershed
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage import segmentation
import itertools
import string
from skimage import morphology
from scipy import ndimage
import matplotlib.widgets as mwidgets
from skimage.morphology import disk
from skimage.filters import rank
# sample = "p41m"
# arrays = ["dif0", "dif1", "dif2", "dif3", "dif4", "dif5", "dif6"]
#arrays = ["dif5"]
#sample = "p52m"
#arrays = ["dif0", "dif1", "dif2", "dif3", "dif5", "dif6"]
#arrays = ["dif0"]

sample = "p45m4"
array = "did5"


#nmpx = 3.100586  # nm/px



#nmpx = 5.17  # nm/px


path = '/home/sei/REM/'+sample+'/'


savedir = path + 'plots/'
denoisedir = path +'denoised/'

show_plots = False

def get_letters(size):
    def iter_all_ascii():
        size = 1
        while True:
            for s in itertools.product(string.ascii_uppercase, repeat=size):
                yield "".join(s)
            size += 1

    letters = np.array([])
    for s in itertools.islice(iter_all_ascii(), size):
        letters = np.append(letters, s)
    return letters


def get_numbers(size):
    def iter_all_numbers():
        size = 1
        while True:
            for s in itertools.product(string.digits[0:10], repeat=size):
                yield "".join(s)
            size += 1

    numbers = np.array([])
    for s in itertools.islice(iter_all_numbers(), size):
        numbers = np.append(numbers, s)
    return numbers

def plot_particles(image1, image2, image3, fname):
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
    ax0, ax1, ax2 = axes
    ax0.imshow(image1, interpolation='nearest')
    # ax0.set_title('Overlapping objects')
    ax1.imshow(image2, cmap=plt.cm.jet, interpolation='nearest')
    # ax1.set_title('Distances')
    ax2.imshow(image3, cmap=plt.cm.spectral, interpolation='nearest')
    # ax2.set_title('Separated objects')
    for ax in axes:
        ax.axis('off')
    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    # plt.show()
    plt.savefig(fname)
    plt.close()


def plot_comparison(original, filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.imshow(original)
    ax1.axis('off')
    ax2.imshow(filtered)
    ax2.axis('off')

class GetXY:

    def __init__(self,img):
        #figWH = (8,5) # in
        self.fig = plt.figure()#figsize=figWH)
        self.ax = self.fig.add_subplot(111)
        self.img = img
        #plt.imshow(img)
        self.ax.imshow(img, cmap="gray")
        self.x =[]
        self.y = []

        self.cursor = mwidgets.Cursor(self.ax, useblit=True, color='Red',linewidth=0.5)
        #self.cursor.horizOn = False

        self.connect = self.ax.figure.canvas.mpl_connect
        self.disconnect = self.ax.figure.canvas.mpl_disconnect

        self.clickCid = self.connect("button_press_event",self.onClick)

    def onClick(self, event):
        if event.button == 1:
            if event.inaxes:
                #ind = self.find_nearest(self.x,event.xdata)
                self.x.append(event.xdata)
                self.y.append(event.ydata)
                self.ax.scatter(event.xdata,event.ydata,color="Red",marker="x",zorder=100,s=50)
                self.fig.canvas.draw()
        else:
            self.cleanup()

    def cleanup(self):
        self.disconnect(self.clickCid)
        plt.close()

try:
    os.mkdir( savedir )
except:
    print("failed to create "+savedir)

try:
    os.mkdir( denoisedir )
except:
    print("failed to create "+denoisedir)

try:
    nmpx = np.loadtxt(path+"nmppx")
except:
    raise RuntimeError("nmppx not found!")

print(nmpx)


files = []
listdir = os.listdir(path)
letters = get_letters(1000)
numbers = get_numbers(1000)
print(listdir)
for l in letters:
    for n in numbers:
        if l + n + '.TIF' in listdir:
            files.append(l + n)

#files = ['E2']

n= len(files)
particles = np.zeros(n)
area = np.zeros(n)
dist = np.zeros(n)
rdiff = np.zeros(n)
r1 = np.zeros(n)
r2 = np.zeros(n)
width = 80

for i,file in enumerate(files):

    fname = path + file + ".TIF"
    data = scipy.misc.imread(fname)
    print("Image Size: " + str(data.shape))
    # data = data[:,:,0]
    data = exposure.rescale_intensity(data)

    sname = denoisedir + file + '_denoised.jpg'
    if False: #os.path.isfile(sname):
        fdata = scipy.misc.imread(sname)
    else:
        fdata = data
        #p1, p2 = np.percentile(data, (2, 99))
        #fdata = exposure.rescale_intensity(data, in_range=(p1, p2))

        #fdata = ndimage.median_filter(data, footprint=morphology.disk(2), mode="mirror")
        #fdata = ndimage.median_filter(fdata, footprint=morphology.disk(2), mode="mirror")
        #fdata = ndimage.median_filter(fdata, footprint=morphology.disk(2), mode="mirror")
        # fdata = morphology.opening(fdata)
        # fdata = denoise_tv_chambolle(fdata,weight=0.05, multichannel=False)
        # fdata = denoise_tv_chambolle(fdata, weight=0.05, multichannel=False)
        # fdata = denoise_tv_chambolle(fdata, weight=0.05, multichannel=False)

        #fdata = denoise_bilateral(data, sigma_color=0.1, sigma_spatial=3, multichannel=False)
        #fdata = denoise_bilateral(fdata, sigma_color=0.1, sigma_spatial=3, multichannel=False)
        #fdata = denoise_bilateral(fdata, sigma_color=0.1, sigma_spatial=3, multichannel=False)
        scipy.misc.imsave(sname, fdata)

    # oname = denoisedir + file + '_opt.jpg'
    # if os.path.isfile(oname):
    #     fdata = scipy.misc.imread(oname)
    # else:
    #     # p1, p99 = np.percentile(data, (1, 99))
    #     # data = exposure.rescale_intensity(data, in_range=(np.min(data)*100, np.max(data)*100))
    #     seed = np.copy(fdata)
    #     seed[1:-1, 1:-1] = fdata.min()
    #     fdata = fdata - reconstruction(seed, fdata, method='dilation')
    #     fdata = filters.gaussian_filter(fdata, sigma=5)
    #     seed = None
    #
    #     seed = np.copy(fdata)
    #     seed[1:-1, 1:-1] = fdata.min()
    #     fdata = fdata - reconstruction(seed, fdata, method='dilation')
    #     seed = None
    #     scipy.misc.imsave(oname, fdata)

    fdata = fdata[150:-150, 150:-150]
    #data = np.flipud(data)
    #fdata = np.flipud(fdata)
    #fdata = exposure.adjust_log(fdata, 1)
    fdata = exposure.adjust_gamma(fdata, 2)

    p1, p2 = np.percentile(fdata, (2, 98))
    fdata = exposure.rescale_intensity(fdata, in_range=(p1, p2))

    XY = GetXY(fdata)
    plt.title(file)
    plt.show()
    XY.cleanup()
    x = XY.x
    y = XY.y

    if len(x) > 1:
        gap = np.sqrt( (x[1]-x[0])**2 + (y[1]-y[0])**2) * nmpx
        print('structure ' + file + ' has 2 blobs with a gap of: ' + str(gap) + ' nm')

    # plt.imshow(fdata, cmap="gray")
    # plt.show()

    if len(x) > 1:
        gap = np.sqrt( (x[1]-x[0])**2 + (y[1]-y[0])**2) * nmpx

        plt.imshow(fdata,cmap=plt.get_cmap('gray'))
        plt.scatter(x,y,color="Red",marker="x",zorder=100,s=50)
        plt.savefig(savedir + file + "_contour.png",dpi=400)
        plt.close()
        dist[i] = np.sqrt( (x[1]-x[0])**2 + (y[1]-y[0])**2) * nmpx
        print('structure ' + file + ' has 2 blobs with a gap of: ' + str(dist[i]) + ' nm')
        particles[i] = 2
    else:
        rdiff[i] = -1.0
        dist[i] = -1.0
        print('structure ' + file + ' is bad  blobs')

    imsave(savedir + file + ".png", fdata)

    # particles[i] = area[i] /
    #plot_particles(data, fdata , labeled, savedir + file + "_detection.png")

# sorted = np.argsort(area)
# area = area[sorted]
# particles = particles[sorted]

ids = np.array(files)

# ids = ids[sorted]
# data = np.append(ids.reshape(ids.shape[0], 1), area.reshape(area.shape[0], 1),1)
# data = np.append(data,particles.reshape(particles.shape[0], 1), 1)

print('-> Writing measured values to file')
f = open(path + "/" + sample + "_" + array + "_particles_SEM.csv", 'w')
f.write("id,area,rdiff,dist,particles,r1,r2" + "\r\n")
for i in range(len(ids)):
    f.write(str(ids[i]) + "," + str(area[i]) + "," + str(rdiff[i]) + "," + str(dist[i]) + "," + str(particles[i])  + "," + str(r1[i])  + "," + str(r2[i]) + "\r\n")

f.close()

data = None
fdata = None
print('-> Processing of ' + path + ' finished')
