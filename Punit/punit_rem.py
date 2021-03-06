__author__ = 'sei'

import os

#from plotsettings import *
import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
#sns.set_context("paper")
#sns.set_style("ticks")

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
from skimage.morphology import reconstruction, label
from skimage.morphology import remove_small_objects
from skimage.morphology import watershed
from skimage.restoration import denoise_bilateral
from skimage.segmentation import relabel_sequential, random_walker, slic
import re
import exifread


nmpx = 500/146  # nm/px


path = '/home/sei/Punit_Report/SEM/'
savedir = path + 'plots/'


try:
    os.mkdir(savedir)
except:
    pass


files = []
for file in os.listdir(path):
    if re.search(r"\.(TIF)$", file) is not None:
        #print(file[-7:-4])
        if file[-7:-4] != '000':
            files.append(file)

print(files)


#file = path+files[0]

# # Open image file for reading (binary mode)
# with open(path+file, 'rb') as f:
#     tags = exifread.process_file(f)
#     for tag in tags.keys():
#         #if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
#         print("Key: %s, value %s" % (tag, tags[tag]))

files = np.array(files)
sort = np.argsort(files)
labels = files[sort]

coverage = np.array([])
n = np.array([])
area_mean = np.array([])
area_err = np.array([])

labels = np.array([])

for file in files:
    print(file)
    labels = np.append(labels, file[:-4])
    file = path+file

    pic = scipy.misc.imread(file)
    #print("Image Size: " + str(pic.shape))
    pic = pic[:420,:]

    # plt.imshow(pic)
    # plt.show()

    pic = exposure.rescale_intensity(pic)

    ffile = file[:-4] + '_denoised.jpg'
    if os.path.isfile(ffile):
        pic = scipy.misc.imread(ffile)
    else:
        pic = denoise_bilateral(pic, sigma_color=0.1, sigma_spatial=3, multichannel=False)
        #pic = denoise_bilateral(pic, sigma_color=0.01, sigma_spatial=1, multichannel=False)
        scipy.misc.imsave(ffile, pic)


    #p1, p99 = np.percentile(pic, (1, 99))
    #pic = exposure.rescale_intensity(pic, in_range=(p1, p99))

    # plt.imshow(pic)
    # plt.show()

    thresh = threshold_otsu(pic)*1.1
    bin = pic > thresh
    #bin = pic > np.max(pic)/2

    # plt.imshow(bin)
    # plt.show()

    c = np.sum(bin) / (pic.shape[0]*pic.shape[1])

    print('coverage:' + str(c))

    coverage = np.append(coverage,c)

    #distance = filters.gaussian_filter(distance, sigma=2)
    # plt.imshow(distance)
    # plt.show()

    #local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((21, 21)),min_distance=30, labels=bin)
    #local_maxi = peak_local_max(distance, indices=False,min_distance=25, labels=bin)
    #markers = measure.label(local_maxi)

    blobs_labels = measure.label(bin)

    plt.imshow(blobs_labels)
    plt.title(file)
    plt.show()

    areas = np.array([])
    for region in regionprops(blobs_labels,intensity_image=pic):
        if region.area > 1:
            # # plt.imshow(region.intensity_image)
            # # plt.show()
            #
            # b = region.image
            # distance = ndimage.distance_transform_edt(b)
            # distance = filters.gaussian_filter(distance, sigma=3)
            # #plt.imshow(distance)
            # #plt.show()
            #
            # local_maxi = peak_local_max(distance, indices=False,footprint=np.ones((3, 3)), labels=b)
            # markers = measure.label(local_maxi)
            #
            # #plt.imshow(markers)
            # #plt.show()
            #
            # if local_maxi.sum() > 1:
            #     #labeled = watershed(-distance, markers, mask=b)
            #     markers[~b] = -1
            #     labeled = random_walker(b,markers,multichannel=False,copy=False,beta=1000)
            #     cmap = matplotlib.colors.ListedColormap(np.random.rand(labeled.max(), 3))
            #     #plt.imshow(labeled,cmap=cmap)
            #     #plt.show()

            #areas = np.append(areas, region.area * nmpx ** 2)
            areas = np.append(areas, region.area * nmpx ** 2)

    #distance = filters.gaussian_filter(distance, sigma=9)
    #labeled = watershed(-distance, markers, mask=bin)
    #markers = label(local_maxi)
    #markers[~bin] = -1
    #labeled = random_walker(bin,markers,multichannel=False,copy=False)
    #labeled = slic(pic, multichannel=False)
    #labeled = remove_small_objects(labeled, 100)
    #labeled = relabel_sequential(labeled)[0]

    n_hist, b_hist, patches = plt.hist(areas, 30, histtype='stepfilled')
    plt.title(file)
    bin_max = np.argmax(n_hist)
    plt.show()

    # areas = np.array([])
    # for region in regionprops(labeled):
    #     areas = np.append(areas, region.area*nmpx**2)
    #     #xy = np.vstack((xy, region.centroid))
    #

    area_mean = np.append(area_mean, b_hist[bin_max])

    n = np.append(n,len(areas))
    #area_mean = np.append(area_mean,np.mean(areas))
    area_err = np.append(area_err,np.std(areas))
    #
    # print(areas.shape)
    # print(np.mean(areas))
    # print(np.std(areas))




sort = np.argsort(labels)
labels = labels[sort]
coverage = coverage[sort]
n = n[sort]
area_mean = area_mean[sort]
area_err = area_err[sort]

print('-> Writing measured values to file')
with open(path + "coverage.csv", 'w') as f:
    f.write("label,coverage,n,area_mean,area_err" + "\r\n")
    for i in range(len(files)):
        f.write( labels[i] + "," + str(coverage[i]) + "," + str(n[i]) + "," + str(area_mean[i]) + "," + str(area_err[i]) + "\r\n")


#
#
#
#
# for array in arrays:
#
#     print('-> Starting with ' + sample + ' ' + array)
#
#     savedir = path + array + '/plots/'
#     fname = path + array + '/' + sample + "_" + array + ".jpg"
#     # fname = path + "test.png"
#
#
#     show_plots = False
#
#
#     def plot_particles(image1, image2, image3, fname):
#         fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
#         ax0, ax1, ax2 = axes
#         ax0.imshow(image1, interpolation='nearest')
#         # ax0.set_title('Overlapping objects')
#         ax1.imshow(image2, cmap=plt.cm.jet, interpolation='nearest')
#         # ax1.set_title('Distances')
#         ax2.imshow(image3, cmap=plt.cm.spectral, interpolation='nearest')
#         # ax2.set_title('Separated objects')
#         for ax in axes:
#             ax.axis('off')
#         fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
#         # plt.show()
#         plt.savefig(fname)
#         plt.close()
#
#
#     def plot_comparison(original, filtered):
#         fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
#         ax1.imshow(original)
#         ax1.axis('off')
#         ax2.imshow(filtered)
#         ax2.axis('off')
#
#
#     try:
#         os.mkdir(path + savedir)
#     except:
#         pass
#
#     print("-> loading and processing sem image")
#
#     data = scipy.misc.imread(fname)
#     print("Image Size: " + str(data.shape))
#     # data = data[:,:,0]
#     data = exposure.rescale_intensity(data)
#
#     sname = fname[:-4] + '_denoised.jpg'
#     if os.path.isfile(sname):
#         fdata = scipy.misc.imread(sname)
#     else:
#         #fdata = denoise_bilateral(data, sigma_color=0.1, sigma_spatial=3, multichannel=False)
#         fdata = denoise_bilateral(data, sigma_color=0.01, sigma_spatial=1, multichannel=False)
#         scipy.misc.imsave(sname, fdata)
#
#     oname = fname[:-4] + '_opt.jpg'
#     if os.path.isfile(oname):
#         fdata = scipy.misc.imread(oname)
#     else:
#         # p1, p99 = np.percentile(data, (1, 99))
#         # data = exposure.rescale_intensity(data, in_range=(np.min(data)*100, np.max(data)*100))
#         seed = np.copy(fdata)
#         seed[1:-1, 1:-1] = fdata.min()
#         fdata = fdata - reconstruction(seed, fdata, method='dilation')
#         fdata = filters.gaussian_filter(fdata, sigma=5)
#         seed = None
#
#         seed = np.copy(fdata)
#         seed[1:-1, 1:-1] = fdata.min()
#         fdata = fdata - reconstruction(seed, fdata, method='dilation')
#         seed = None
#         scipy.misc.imsave(oname, fdata)
#
#     data = np.flipud(data)
#     fdata = np.flipud(fdata)
#
#     if show_plots:
#         plt.imshow(fdata, cmap="gray")
#         plt.show()
#
#     thresh = threshold_otsu(fdata)
#     bin = fdata > thresh
#
#     print("-> searching blobs")
#
#     labeled, n = ndimage.label(bin)
#     # labeled = remove_small_objects(labeled, 400)
#
#     # plt.imshow(labeled)
#     # plt.savefig(savedir+"blobdetect.png")
#     # plt.close()
#
#     # xy = np.array(ndimage.center_of_mass(fdata, labeled, range(1, n + 1)))
#     # xy = xy[:,[1,0]]
#
#     xy = np.zeros((0, 2))
#     for region in regionprops(labeled):
#         if region.area > 400:
#             xy = np.vstack((xy, region.centroid))
#
#     xy = xy[:, [1, 0]]
#
#     if show_plots:
#         plt.imshow(labeled)
#         plt.plot(xy[:, 0], xy[:, 1], "rx")
#         plt.show()
#
#     x = []
#     y = []
#     low_dist = False
#     low_ind = -1
#     for i in range(xy.shape[0]):
#         if xy[i, 0] > 0:
#             for j in range(xy.shape[0]):
#                 if (i != j):
#                     d = np.sqrt((xy[i, 0] - xy[j, 0]) ** 2 + (xy[i, 1] - xy[j, 1]) ** 2)
#                     if d < 150:
#                         low_dist = True
#                         low_ind = j
#             if low_dist:
#                 x = np.append(x, xy[i, 0] - (xy[i, 0] - xy[low_ind, 0]) / 2)
#                 y = np.append(y, xy[i, 1] - (xy[i, 1] - xy[low_ind, 1]) / 2)
#                 xy[low_ind, :] = 0
#                 low_dist = False
#             else:
#                 x = np.append(x, xy[i, 0])
#                 y = np.append(y, xy[i, 1])
#
#     xy = np.transpose(np.array([x, y]))
#     print(str(len(xy)) + ' valid blobs found')
#
#     print("-> fitting grid")
#     inds, nxy, ids = fit_grid(xy, nx, ny)
#
#     print(nxy.shape)
#
#     n = nxy.shape[0]
#
#     print("-> plotting "+str(n)+" individual structures")
#     try:
#         os.mkdir(savedir)
#     except:
#         pass
#     particles = np.zeros(n)
#     area = np.zeros(n)
#     dist = np.zeros(n)
#     rdiff = np.zeros(n)
#     width = 80
#
#     for i in range(n):
#         x = nxy[i, 0]
#         y = nxy[i, 1]
#         x = int(x)
#         y = int(y)
#         if x - width > 0:
#             xstart = x - width
#         else:
#             xstart = 0
#
#         if x + width < data.shape[1] - 1:
#             xstop = x + width
#         else:
#             xstop = data.shape[1] - 1
#
#         if y - width > 0:
#             ystart = y - width
#         else:
#             ystart = 0
#
#         if y + width < data.shape[0] - 1:
#             ystop = y + width
#         else:
#             ystop = data.shape[0] - 1
#
#         sub = data[ystart:ystop, xstart:xstop]
#         #sub2 = data[ystart:ystop, xstart:xstop]
#         sub = denoise_bilateral(sub, sigma_color=0.1, sigma_spatial=15, multichannel=False)
#
#         buf = np.zeros((2 * width, 2 * width))
#         buf[:sub.shape[0], :sub.shape[1]] = sub
#
#         thresh = threshold_otsu(sub) * 1.15
#         bin = sub > thresh
#         # area[i] = np.sum(bin)
#
#         labeled, n = ndimage.label(bin)
#
#         thresh = threshold_otsu(sub) * 1.3
#         bin = sub > thresh
#         # bin = convex_hull_image(bin)
#         bin = ndimage.binary_fill_holes(bin)
#         sub = buf
#         sub = np.flipud(sub)
#
#         p2, p98 = np.percentile(sub, (2, 100))
#         sub = exposure.rescale_intensity(sub, in_range=(p2, p98))
#         # labeled, n = ndimage.label(bin)
#         distance = ndimage.distance_transform_edt(bin)
#         distance = filters.gaussian_filter(distance, sigma=5)
#         local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=bin)
#         markers = measure.label(local_maxi)
#         labeled = watershed(-distance, markers, mask=bin)
#         # markers[~bin] = -1
#         # labeled = segmentation.random_walker(bin,markers)
#         labeled = remove_small_objects(labeled, 100)
#         labeled = relabel_sequential(labeled)[0]
#
#         xy = np.zeros((0, 2))
#         areas = np.zeros((0, 1))
#         for region in regionprops(labeled):
#             if region.area > 100:
#                 area[i] += region.area
#                 xy = np.vstack((xy, region.centroid))
#                 areas = np.vstack((areas, region.area*nmpx**2))
#
#         num = xy.shape[0]
#
#         imsave(savedir + ids[i] + ".png", sub)
#
#         particles[i] = num
#
#         if num == 2:
#             if areas[0] > areas[1]:
#                 r1 = np.sqrt(areas[0]/np.pi)
#                 r2 = np.sqrt(areas[1]/np.pi)
#             else:
#                 r1 = np.sqrt(areas[1]/np.pi)
#                 r2 = np.sqrt(areas[0]/np.pi)
#
#             rdiff[i] = (r1 - r2) / r1
#
#             c1 = measure.find_contours(labeled == 2, 0)[0]
#             c2 = measure.find_contours(labeled == 1, 0)[0]
#
#             plt.plot(c1[:, 0], c1[:, 1])
#             plt.plot(c2[:, 0], c2[:, 1])
#             plt.savefig(savedir + ids[i] + "_contour.pdf")
#             plt.close()
#
#             d = np.array([])
#             for k in range(c1.shape[0]):
#                 for l in range(c2.shape[0]):
#                     d = np.append(d, np.sqrt(
#                         np.power(np.real(c1[k, 0] - c2[l, 0]), 2) + np.power(np.real(c1[k, 1] - c2[l, 1]), 2)))
#             dist[i] = np.min(d) * nmpx
#             print('structure ' + ids[i] + ' has 2 blobs with a gap of: ' + str(dist[i]) + ' nm')
#         else:
#             rdiff[i] = -1.0
#             dist[i] = -1.0
#             print('structure ' + ids[i] + ' has ' + str(num) + ' blobs')
#
#         # particles[i] = area[i] /
#         plot_particles(sub, distance, labeled, savedir + ids[i] + "_detection.pdf")
#
#     # sorted = np.argsort(area)
#     # area = area[sorted]
#     # particles = particles[sorted]
#
#     ids = np.array(ids)
#
#     # ids = ids[sorted]
#     # data = np.append(ids.reshape(ids.shape[0], 1), area.reshape(area.shape[0], 1),1)
#     # data = np.append(data,particles.reshape(particles.shape[0], 1), 1)
#
#     print('-> Writing measured values to file')
#     f = open(path + array + "/" + sample + "_" + array + "_particles_SEM.csv", 'w')
#     f.write("id,area,rdiff,dist,particles" + "\r\n")
#     for i in range(len(ids)):
#         f.write(str(ids[i]) + "," + str(area[i]) + "," + str(rdiff[i]) + "," + str(dist[i]) + "," + str(particles[i]) + "\r\n")
#
#     f.close()
#
#     data = None
#     fdata = None
#     print('-> Processing of ' + array + ' finished')
