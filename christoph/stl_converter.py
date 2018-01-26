import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from numba import jit

# Using an existing stl file:
your_mesh = mesh.Mesh.from_file('kegeleinschnitt.stl')


# figure = plt.figure()
# axes = mplot3d.Axes3D(figure)
#
# # Load the STL files and add the vectors to the plot
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
#
# # Auto scale to the mesh size
# scale = your_mesh.points.flatten(-1)
# axes.auto_scale_xyz(scale, scale, scale)
#
# # Show the plot to the screen
# plt.show()





vectors = your_mesh.vectors

print(vectors.shape)
print(vectors[0,:,:])

x = vectors[:,:,0].ravel()
z = vectors[:,:,1].ravel()
y = vectors[:,:,2].ravel()

z = z-z.min()
z = z/z.max()


@jit(nopython=True,parallel=True)
def check_for_doubles(x,y,z):
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                if x[i] == x[j] and y[i] == y[j]:
                    if z[i]>=z[j]:
                        z[j]=-1
                    else:
                        z[i]=-1
    return z

#z = check_for_doubles(x,y,z)
#z = z - 0.0001

inds = z > 0
x = x[inds]
y = y[inds]
z = z[inds]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
plt.show()


xi = np.linspace(x.min()*1.1,x.max()*1.1,1000)
yi = np.linspace(y.min()*1.1,y.max()*1.1,1000)
#zi = griddata(x, y, z, xi, yi)
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='nearest',fill_value=0.0)

plt.imshow(zi,cmap=plt.cm.get_cmap("Greys"))
plt.show()

