__author__ = 'sei'

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from skimage import exposure
from skimage import measure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_minimum
from skimage.io import imsave
from skimage.measure import regionprops
from skimage.morphology import reconstruction
from skimage.morphology import remove_small_objects
from skimage.morphology import watershed
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, deconvolution
from skimage import segmentation
import itertools
import string
from skimage import morphology
from scipy import ndimage
from numba import jit
from matplotlib.widgets import Slider, Button
from scipy.signal import convolve2d as conv2

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

@jit(nopython=True)
def find_shortest_distance(c1,c2):
    d = np.zeros((c1.shape[0],c2.shape[0]))
    k = 0
    l = 0
    for k in range(c1.shape[0]):
        for l in range(c2.shape[0]):
            d[k,l] = np.sqrt(np.power(c1[k, 0] - c2[l, 0], 2) + np.power(c1[k, 1] - c2[l, 1], 2))
    return d.min()


def find_particles(bin):

    labeled, n = ndimage.label(bin)

    num = 0
    for region in regionprops(labeled):
        if region.area > 500:
            num += 1

    if num == 2:
        c1 = measure.find_contours(labeled == 2, 0)[0]
        c2 = measure.find_contours(labeled == 1, 0)[0]
        d = find_shortest_distance(c1, c2)
    elif num == 1:
        c1 = measure.find_contours(labeled == 1, 0)[0]
        c2 = None
        d = 0
    else:
        c1 = None
        c2 = None
        d = 0

    return d,c1,c2

def gauss2D(pos, amplitude, xo, yo, fwhm, offset):
    sigma = fwhm / 2.3548
    g = offset + amplitude * np.exp(
        -(np.power(pos[0] - xo, 2.) + np.power(pos[1] - yo, 2.)) / (2 * np.power(sigma, 2.)))
    return g.ravel()

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
    #data = exposure.rescale_intensity(data)
    data = np.array(data,dtype=np.float)
    data -= data.min()
    data /= data.max()
    data += .1

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
        fdata = denoise_tv_chambolle(fdata,weight=0.05, multichannel=False)
        # fdata = denoise_tv_chambolle(fdata, weight=0.05, multichannel=False)
        # fdata = denoise_tv_chambolle(fdata, weight=0.05, multichannel=False)

        #fdata = denoise_bilateral(data, sigma_color=0.1, sigma_spatial=3, multichannel=False)
        #fdata = denoise_bilateral(fdata, sigma_color=0.1, sigma_spatial=3, multichannel=False)
        #fdata = denoise_bilateral(fdata, sigma_color=0.1, sigma_spatial=3, multichannel=False)
        scipy.misc.imsave(sname, fdata)



    # posx = np.arange(0,10,1)
    # posy = np.arange(0, 10, 1)
    # pos = np.meshgrid(posx,posy)
    # psf = gauss2D(pos,10,np.mean(posx),np.mean(posy),2,0)
    # psf = np.reshape(psf,(len(posx),len(posy)))
    # data = deconvolution.richardson_lucy(data, psf, iterations=3)


    fdata = exposure.adjust_gamma(fdata, 5)
    # p1, p2 = np.percentile(fdata, (1, 99))
    # fdata = exposure.rescale_intensity(fdata, in_range=(p1, p2))

    fdata -= fdata.min()
    fdata /= fdata.max()

    fdata_buf = fdata[120:-150,200:-150]
    data_buf = data[120:-150, 200:-150]
    data_buf = exposure.adjust_gamma(data_buf, 5)
    #p1, p2 = np.percentile(data_buf, (5, 98))
    #data_buf = exposure.rescale_intensity(data_buf, in_range=(p1, p2))

    fig, ax = plt.subplots()
    plt.imshow(data_buf,cmap="Greys_r")
    plt.subplots_adjust(left=0.25, bottom=0.25)
    thresh0 = threshold_minimum(fdata)*1.5
    bin = fdata_buf > thresh0
    #labeled, n = ndimage.label(bin)
    #img = plt.imshow(labeled)
    d, c1, c2 = find_particles(bin)
    if c1 is not None:
        lines1, = plt.plot(c1[:, 1], c1[:, 0])
    else:
        lines1, = plt.plot(0, 0)
    plt.setp(lines1, color='b', linewidth=0.5)
    if c2 is not None:
        lines2, = plt.plot(c2[:, 1], c2[:, 0])
    else:
        lines2, = plt.plot(0, 0)
    plt.setp(lines2, color='b', linewidth=0.5)
    ax.set_title(str(d))

    axcolor = 'lightgoldenrodyellow'
    #axthresh = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axthresh = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sthresh = Slider(axthresh, 'Threshold', 0.0, 1.0, valinit=thresh0)


    def update(val):
        thresh = sthresh.val
        bin = fdata_buf > thresh
        d, c1, c2 = find_particles(bin)
        ax.set_title(str(d))
        if c1 is not None:
            lines1.set_xdata(c1[:, 1])
            lines1.set_ydata(c1[:, 0])
        else:
            lines1.set_xdata(0)
            lines1.set_ydata(0)
        if c2 is not None:
            lines2.set_xdata(c2[:, 1])
            lines2.set_ydata(c2[:, 0])
        else:
            lines2.set_xdata(0)
            lines2.set_ydata(0)
        # labeled, n = ndimage.label(bin)
        # img.set_data(labeled)

        fig.canvas.draw_idle()

    sthresh.on_changed(update)

    plt.show()

    thresh = sthresh.val
    bin = fdata > thresh
    labeled, n = ndimage.label(bin)

    xy = np.zeros((0, 2))
    areas = np.zeros((0, 1))
    for region in regionprops(labeled):
        if region.area > 100:
            xy = np.vstack((xy, region.centroid))
            areas = np.vstack((areas, region.area*nmpx**2))

    num = xy.shape[0]
    particles[i] = num

    if num == 2:
        c1 = measure.find_contours(labeled == 2, 0)[0]
        c2 = measure.find_contours(labeled == 1, 0)[0]
        d = find_shortest_distance(c1,c2)


    fig = plt.imshow(labeled)
    plt.set_cmap('hot')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(denoisedir+file+"_blobdetect.png", bbox_inches='tight', pad_inches=-0.1)
    plt.close()

    if num == 2:
        if areas[0] > areas[1]:
            r1[i] = np.sqrt(areas[0]/np.pi)
            r2[i] = np.sqrt(areas[1]/np.pi)
        else:
            r1[i] = np.sqrt(areas[1]/np.pi)
            r2[i] = np.sqrt(areas[0]/np.pi)

        rdiff[i] = (r1[i] - r2[i]) / r1[i]

        data_buf = exposure.adjust_gamma(data, 5)
        p1, p2 = np.percentile(data_buf, (5, 98))
        data_buf = exposure.rescale_intensity(data_buf, in_range=(p1, p2))

        #fig = plt.figure()
        fig = plt.imshow(data_buf,cmap=plt.get_cmap('Greys_r'))
        lines1 = plt.plot(c1[:, 1],c1[:, 0])
        plt.setp(lines1, color='b', linewidth=0.5)
        lines2 = plt.plot(c2[:, 1], c2[:, 0])
        plt.setp(lines2, color='b', linewidth=0.5)

        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(savedir + file + "_contour.png",dpi=400, bbox_inches='tight', pad_inches=-0.1)
        plt.close()


        dist[i] = d * nmpx
        print('structure ' + file + ' has 2 blobs with a gap of: ' + str(dist[i]) + ' nm')
    else:
        rdiff[i] = -1.0
        dist[i] = -1.0
        print('structure ' + file + ' has ' + str(num) + ' blobs')

    bin = fdata > thresh*0.8
    area[i] = np.sum(bin)*nmpx**2

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
