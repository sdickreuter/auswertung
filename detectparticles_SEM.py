__author__ = 'sei'

import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

from skimage import data
from skimage.io import imsave
from skimage.feature import peak_local_max, blob_log, blob_doh, blob_dog
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage import exposure
from skimage.restoration import nl_means_denoising
from skimage.morphology import disk
from skimage.morphology import erosion
from skimage.filters import rank

from skimage.filters import sobel
from skimage.segmentation import slic, join_segmentations
from skimage.color import label2rgb

#path = '/home/sei/Auswertung/2C2/2C2_75hept_B2/'
#path = '/home/sei/Auswertung/2C2/2C2_150hex_C2/'
#path = '/home/sei/Auswertung/2C2/2C2_150tri_A1/'
#path = '/home/sei/Auswertung/2C2/2C2_200hex_B1/'
#path = '/home/sei/Auswertung/2C2/2C2_200tri_A3/'

#path = '/home/sei/Auswertung/2C1/2C1_150hept_B2/'
#path = '/home/sei/Auswertung/2C1/2C1_150hex_B3/'
#path = '/home/sei/Auswertung/2C1/2C1_200hex_A2/'
path = '/home/sei/Auswertung/2C1/2C1_75hept_B2/'
#path = '/home/sei/Auswertung/2C1/2C1_100hex_C2/'

#path = '/home/sei/Auswertung/rods5/'

#path = '/home/sei/REM/8/'


savedir = 'plots/'

#fname = path + "B2_Q01 - B2_Q04.jpg"
fname = path + "sem.jpg"
#fname = path + "8_C3.png"

#grid dimensions
nx = 7
ny = 7


def plot_particles(image1,image2,image3, fname):
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
    ax0, ax1, ax2 = axes
    ax0.imshow(image1, interpolation='nearest')
    ax0.set_title('Overlapping objects')
    ax1.imshow(image2, cmap=plt.cm.jet, interpolation='nearest')
    ax1.set_title('Distances')
    ax2.imshow(image3, cmap=plt.cm.spectral, interpolation='nearest')
    ax2.set_title('Separated objects')
    for ax in axes:
        ax.axis('off')
    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                        right=1)
    #plt.show()
    plt.savefig(fname, format='png')
    plt.close()

def plot_comparison(original, filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.imshow(original)
    ax1.axis('off')
    ax2.imshow(filtered)
    ax2.axis('off')

try:
    os.mkdir(path+savedir)
except:
    pass

data = scipy.misc.imread(fname)
#data = filters.gaussian_filter(data, sigma=1)
#data = denoise_bilateral(data, sigma_range=0.5, sigma_spatial=3,multichannel=False)
data = denoise_tv_chambolle(data, weight=0.05, multichannel=False)
#p2, p98 = np.percentile(data, (2, 98))
#img_rescale = exposure.rescale_intensity(data, in_range=(p2, p98))

selem = disk(15)
#data = rank.equalize(data, selem=selem)
#data = rank.otsu(data, selem)

data = exposure.rescale_intensity(data)
#data = exposure.adjust_gamma(data, 5)

plt.imshow(data, cmap="gray")
plt.show()


fdata = filters.gaussian_filter(data, sigma=10)
thresh = threshold_otsu(data)/3

plt.imshow(fdata > thresh)
plt.show()

labeled, n = ndimage.label(fdata > thresh)
xy = np.array(ndimage.center_of_mass(fdata, labeled, range(1, n + 1)))

xy = xy[:,[1,0]]

plt.imshow(labeled)
plt.plot(xy[:,0],xy[:,1],"rx")
plt.show()


# plt.imshow(genmask((80,80),5,sub))
# plt.show()
#
# n = 4
# r = 5.5
# pos = np.zeros((n*2),dtype=np.int32)
# for i in range(n):
#     pos[i] = sub.shape[0]/2
#     pos[n+i] = sub.shape[1]/2
#
# # error function for minimizing
# def calc_error(params):
#     buf = sub.copy()
#     overlap = np.ones(sub.shape)
#     for i in range(n):
#         mask = genmask((params[i],params[n+i]),r,buf)
#         buf[mask] = 0
#         overlap[mask] *= 2
#     error = np.sum(buf)
#     error += np.sum(overlap)/500
#     return error
#
# start = pos.copy()
# #res = minimize(calc_error, start, method='L-BFGS-B', jac=False, options={'disp': True, 'maxiter': 500})
# res = minimize(calc_error, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
# #bnds = ((0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100))
# #res = minimize(calc_error, start, method='SLSQP',bounds=bnds,tol=1e-12, options={ 'disp': True, 'maxiter': 500})
#
#
# buf = sub.copy()
# for i in range(n):
#     mask = genmask((res.x[i],res.x[n+i]),r,buf)
#     buf[mask] = 1
#
# plt.imshow(buf)
# plt.show()



#
# #blobs_log = blob_log(data, max_sigma=30, num_sigma=10, threshold=.1)
# blobs_log = blob_log(sub, max_sigma=7, num_sigma=50, threshold=thresh/2)
#
# fig, ax = plt.subplots(1, 1)
# ax.set_title("blobs")
# ax.imshow(sub, interpolation='nearest')
# for blob in blobs_log:
#     y, x, r = blob
#     c = plt.Circle((x, y), r, color="yellow", linewidth=2, fill=False)
#     ax.add_patch(c)
#
# plt.show()
#
#
#
#
#
# edges = sobel(test)
# markers = np.zeros_like(test)
# foreground, background = 1, 2
# markers[test < 30.0 / 255] = background
# markers[test > 150.0 / 255] = foreground
#
# ws = watershed(edges, markers)
# seg1 = ndimage.label(ws == foreground)[0]
#
#
# # make segmentation using SLIC superpixels
# seg2 = slic(test, n_segments=117, max_iter=160, sigma=1, compactness=0.75,
#             multichannel=False)
#
# # combine the two
# segj = join_segmentations(seg1, seg2)
#
# # show the segmentations
# fig, axes = plt.subplots(ncols=4, figsize=(9, 2.5))
# axes[0].imshow(test, cmap=plt.cm.gray, interpolation='nearest')
# axes[0].set_title('Image')
#
# color1 = label2rgb(seg1, image=test, bg_label=0)
# axes[1].imshow(color1, interpolation='nearest')
# axes[1].set_title('Sobel+Watershed')
#
# color2 = label2rgb(seg2, image=test, image_alpha=0.5)
# axes[2].imshow(color2, interpolation='nearest')
# axes[2].set_title('SLIC superpixels')
#
# color3 = label2rgb(segj, image=test, image_alpha=0.5)
# axes[3].imshow(color3, interpolation='nearest')
# axes[3].set_title('Join')
#
# for ax in axes:
#     ax.axis('off')
# fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
# plt.show()


# calculate grid points
def make_grid(nx, ny, x0, y0, ax, ay, bx, by):
    letters = [chr(c) for c in range(65, 91)]
    a0 = np.array([x0, y0])
    a = np.array([ax, ay])
    b = np.array([bx, by])
    points = np.zeros([nx * ny, 2])
    ids = np.empty(nx * ny, dtype=object)
    for i in range(nx):
        for j in range(ny):
            #print(j*ny+i)
            points[i + j * ny, :] = a0 + a * i + b * j
            ids[i + j * ny] = (letters[nx - i - 1] + "{0:d}".format(ny - j))
    ordered = np.argsort(points[:, 0])
    points = points[ordered, :]
    points[:, :] = points[::-1, :]
    ids = ids[ordered]
    return points, ids


#calculate min dist between sets of points
def calc_mindists(points1, points2):
    dists = np.zeros(points1.shape[0])
    indices = np.zeros(points1.shape[0], dtype=np.int)
    buf = np.zeros(points2.shape[0])
    weights = np.zeros(points2.shape[0])
    for i in range(points1.shape[0]):
        for j in range(points2.shape[0]):
            #buf[j] = np.sqrt( np.sum( np.square( points2[j,:] - points1[i,:] ) ) )
            buf[j] = np.sum(np.square(points2[j, :] - points1[i, :]))
        indices[i] = np.argmin(buf)
        weights[indices[i]] += 1;
        dists[i] = buf[indices[i]]*buf[indices[i]] * weights[indices[i]]
    return dists, indices

# function for adding up distances of points between two grids
def grid_diff(points1, points2):
    return np.sum(calc_mindists(points1, points2))

#min_ind = np.argmin(xy[:, 1] * xy[:, 0])
#max_ind = np.argmax(xy[:, 1] * xy[:, 0])
#x0 = 0 #xy[min_ind, 0]
#y0 = 0 #xy[min_ind, 1]
#ax = 0.5*(xy[max_ind, 0] - x0) / (nx - 1)
#ay = 0.5*(xy[min_ind, 1] - y0) / (ny - 1)
#bx = 0.5*(xy[min_ind, 0] - x0) / (nx - 1)
#by = 0.5*(xy[max_ind, 1] - y0) / (ny - 1)

min_ind = np.argmin(xy[:, 1] * xy[:, 0])
max_ind = np.argmax(xy[:, 1] * xy[:, 0])
x0 = xy[min_ind, 0]
y0 = xy[min_ind, 1]
ax =0.7 * (xy[max_ind, 0] - x0) / (nx - 1)
ay =0.7 * (xy[min_ind, 1] - y0) / (ny - 1)
bx =0.7 * (xy[min_ind, 0] - x0) / (nx - 1)
by =0.7 * (xy[max_ind, 1] - y0) / (ny - 1)

# error function for minimizing
def calc_error(params):
    grid, ids = make_grid(nx, ny, params[0], params[1], params[2], params[3], params[4], params[5])
    #grid, ids = make_grid(nx,ny,x0,y0,params[0],params[1],params[2],params[3])
    return grid_diff(xy, grid)


start = np.array([x0, y0, ax, ay, bx, by])
bnds = ((0, 2*x0), (0, 2*y0), (0, ax*nx), (-by*ny, by*ny), (-ax*nx, ax*nx), (0, by*ny))

res = minimize(calc_error, start, method='L-BFGS-B', jac=False, options={'disp': True, 'maxiter': 500})
#res = minimize(calc_error, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
#res = minimize(calc_error, start, method='SLSQP',tol=1e-12, options={ 'disp': True, 'maxiter': 500})


grid, ids = make_grid(nx, ny, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5])
d, inds = calc_mindists(grid,xy)

xy = xy[inds, :]

plt.plot(grid[:,0],grid[:,1],"bx")
plt.plot(xy[:,0],xy[:,1],"r.")
plt.show()

validpoints = np.where(d < 7e7)[0]
ids = ids[validpoints]
grid = grid[validpoints, :]
xy = xy[validpoints, :]

plt.figure(figsize=(5,5))
plt.imshow(data, cmap="gray_r" )
#plt.plot(grid[:,0],grid[:,1],"bx")
#plt.plot(xy[:,0],xy[:,1],"r.")
for x, y, s in zip(xy[:,0],xy[:,1],ids):
    plt.text(x-20,y-50,s)

#plt.show()
plt.savefig(path+"grid.pdf", format='pdf')
plt.close()

n = xy.shape[0]

particles = np.zeros(n)
area = np.zeros(n)
width = 30
nxy = [[int(x),int(y)] for x,y in xy]
for i in range(n):
    x,y = nxy[i]
    if x - width > 0:
        xstart = x - width
    else:
        xstart = 0
    if x + width < data.shape[1]-1:
        xstop = x + width
    else:
        xstop = data.shape[1]-1
    if y - width > 0:
        ystart = y - width
    else:
        ystart = 0
    if y + width < data.shape[0]-1:
        ystop = y + width
    else:
        ystop = data.shape[0]-1
    sub = data[ystart:ystop,xstart:xstop]

    buf = np.zeros((2*width,2*width))
    buf[:sub.shape[0],:sub.shape[1]] = sub

    thresh = threshold_otsu(sub) / 2
    area[i] = np.sum(sub > thresh)

    sub = buf

    #thresh = threshold_otsu(sub)*1.5

    #p2, p98 = np.percentile(sub, (2, 98))
    #sub = exposure.rescale_intensity(sub, in_range=(p2, p98))


    #plt.imshow(sub, cmap="gray")
    #plt.axis('off')
    #plt.savefig(path+savedir+ids[i]+".png")
    #plt.close()
    imsave(path+savedir+ids[i]+".png",sub)
    #fimage = filters.gaussian_filter(sub, sigma=1)
    image = buf > thresh
    #selem = disk(3)
    #image = erosion(image, selem)
#
# #    blobs = blob_log(sub, max_sigma=6.5, min_sigma=5, num_sigma=100,overlap=0.7)
#     blobs = blob_doh(sub, max_sigma=6.5, min_sigma=4.5, num_sigma=100,overlap=0.5,threshold=.015)
# #    blobs = blob_dog(sub, max_sigma=7, min_sigma=4.5, overlap=0.4)
#     print(blobs)
#
#     fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
#     ax0, ax1, ax2 = axes
#     ax0.imshow(sub, interpolation='nearest')
#     ax1.imshow(image, cmap=plt.cm.jet, interpolation='nearest')
#     ax2.imshow(sub, cmap="gray", interpolation='nearest')
#     for blob in blobs:
#         y, x, r = blob
#         c = plt.Circle((x, y), r, color="yellow", linewidth=2, fill=False)
#         ax2.add_patch(c)
#     for ax in axes:
#         ax.axis('off')
#     fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
#                         right=1)
#     #plt.show()
#     plt.savefig(path+savedir+ids[i]+"_detection.png", format='png')
#     plt.close()

    # Generate the markers as local maxima of the distance to the background
    distance = ndimage.distance_transform_edt(image)
    distance = filters.gaussian_filter(distance, sigma=2)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((7, 7)), labels=image)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    particles[i] = np.max(labels)
    #particles[i] = area[i] /
    plot_particles(image,-distance,labels,path+savedir+ids[i]+"_detection.png")



#sorted = np.argsort(area)
#area = area[sorted]
#particles = particles[sorted]

ids = np.array(ids)
#ids = ids[sorted]
data = np.append(ids.reshape(ids.shape[0], 1), area.reshape(area.shape[0], 1),1)
data = np.append(data,particles.reshape(particles.shape[0], 1), 1)

f = open(path+"particles_SEM.csv", 'w')
f.write("id,area,particles"+"\r\n")
for i in range(len(ids)):
    f.write(str(ids[i])+","+str(area[i])+","+str(particles[i])+"\r\n")

f.close()

