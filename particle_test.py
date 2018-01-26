import numpy as np
import scipy
import matplotlib.pyplot as plt

from skimage.morphology import watershed
from skimage.feature import peak_local_max
import scipy.ndimage.filters as filters
from skimage.filter import threshold_otsu
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

path = '/home/sei/Auswertung/2C1/2C1_75hept_B2/'

image = scipy.misc.imread(path+"A1.png")

template = scipy.misc.imread(path+"template.png")

thresh = threshold_otsu(image)
print(thresh)
image = image[:,:,0]

selem = disk(6)
image = erosion(image, selem)

#fimage = filters.gaussian_filter(image, sigma=1)
image = image > thresh

plt.imshow(image)
plt.show()

distance = scipy.ndimage.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((11, 11)),labels=image)
markers = scipy.ndimage.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
ax0, ax1, ax2 = axes

ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title('Overlapping objects')
ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
ax1.set_title('Distances')
ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
ax2.set_title('Separated objects')

for ax in axes:
    ax.axis('off')

fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)
plt.show()