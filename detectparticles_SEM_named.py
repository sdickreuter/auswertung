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
from skimage.filters import threshold_otsu,threshold_minimum
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
        fdata = morphology.opening(fdata)
        fdata = denoise_tv_chambolle(fdata,weight=0.05, multichannel=False)
        fdata = denoise_tv_chambolle(fdata, weight=0.05, multichannel=False)
        fdata = denoise_tv_chambolle(fdata, weight=0.05, multichannel=False)

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

    #data = np.flipud(data)
    #fdata = np.flipud(fdata)
    fdata = exposure.adjust_gamma(fdata, 5)
    #
    p1, p2 = np.percentile(fdata, (2, 98))
    fdata = exposure.rescale_intensity(fdata, in_range=(p1, p2))

    fdata -= fdata.min()
    fdata /= fdata.max()

    if show_plots:
        plt.imshow(fdata, cmap="gray")
        plt.show()

    #thresh = threshold_otsu(fdata)
    thresh = threshold_minimum(fdata)


    #bin = fdata > thresh*1.70
    #bin = fdata > thresh*1.7
    bin = fdata > thresh * 1.5
    print('Threshold:' + str(thresh))


    #bin = fdata > 0.8

    #bin = fdata > fdata.max()/2

    #thresh = threshold_otsu(fdata)
    #bin = morphology.dilation(fdata > thresh*1.70)
    #data = morphology.closing(data,morphology.disk(5) )
    #bin = data > thresh*1.65


    print("-> searching blobs")

    labeled, n = ndimage.label(bin)
    #labeled = remove_small_objects(labeled, 100)


    fig = plt.figure()
    #fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(labeled)
    plt.tight_layout()
    plt.savefig(denoisedir+file+"_blobdetect.png")


    # plt.imshow(labeled)
    # plt.savefig(denoisedir+file+"_blobdetect.png")
    # plt.close()
    #imsave(denoisedir+file+"_blobdetect.png",labeled)

    xy = np.zeros((0, 2))
    areas = np.zeros((0, 1))
    for region in regionprops(labeled):
        if region.area > 100:
            xy = np.vstack((xy, region.centroid))
            areas = np.vstack((areas, region.area*nmpx**2))

    num = xy.shape[0]
    particles[i] = num

    if num == 2:
        if areas[0] > areas[1]:
            r1[i] = np.sqrt(areas[0]/np.pi)
            r2[i] = np.sqrt(areas[1]/np.pi)
        else:
            r1[i] = np.sqrt(areas[1]/np.pi)
            r2[i] = np.sqrt(areas[0]/np.pi)

        rdiff[i] = (r1[i] - r2[i]) / r1[i]

        c1 = measure.find_contours(labeled == 2, 0)[0]
        c2 = measure.find_contours(labeled == 1, 0)[0]


        plt.imshow(fdata,cmap=plt.get_cmap('gray'))
        #plt.imshow(labeled)
        lines1 = plt.plot(c1[:, 1],c1[:, 0])
        plt.setp(lines1, color='b', linewidth=0.5)
        lines2 = plt.plot(c2[:, 1], c2[:, 0])
        plt.setp(lines2, color='b', linewidth=0.5)
        plt.savefig(savedir + file + "_contour.png",dpi=400)
        plt.close()

        d = np.array([])
        for k in range(c1.shape[0]):
            for l in range(c2.shape[0]):
                d = np.append(d, np.sqrt(
                    np.power(np.real(c1[k, 0] - c2[l, 0]), 2) + np.power(np.real(c1[k, 1] - c2[l, 1]), 2)))
        dist[i] = np.min(d) * nmpx
        print('structure ' + file + ' has 2 blobs with a gap of: ' + str(dist[i]) + ' nm')
    else:
        rdiff[i] = -1.0
        dist[i] = -1.0
        print('structure ' + file + ' has ' + str(num) + ' blobs')

    #bin = fdata > fdata.max()*0.8
    bin = fdata > thresh * 1.5
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
