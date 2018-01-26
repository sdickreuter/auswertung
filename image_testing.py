import numpy as np
import matplotlib.pyplot as plt

from skimage import color, data, restoration, filters, feature, segmentation, img_as_float
import scipy


sample = "p45m4"
array = "did5"


#nmpx = 3.100586  # nm/px



#nmpx = 5.17  # nm/px


path = '/home/sei/REM/'+sample+'/'
#fname = path + "B1.TIF"
fname = path + "denoised/B1_denoised.jpg"
data = scipy.misc.imread(fname)

im = data
# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=4)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()