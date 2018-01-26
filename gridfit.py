import os
import re
import sys

from plotsettings import *
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

savedir = "/home/sei/"


def minimum_bounding_rectangle(points):
    """
    From Jesse Buesking:
    http://stackoverflow.com/questions/13542855/python-help-to-implement-an-algorithm-to-find-the-minimum-area-rectangle-for-gi/33619018#33619018

    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    #     rotations = np.vstack([
    #         np.cos(angles),
    #         -np.sin(angles),
    #         np.sin(angles),
    #         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def fit_grid_spectra(path, nx, ny, switch_xy=False, flip_x=False, flip_y=False):
    savedir = path

    files = []
    for file in os.listdir(savedir):
        if re.fullmatch(r"([0-9]{5})(.csv)$", file) is not None:
            files.append(file)

    n = len(files)
    xy = np.zeros([n, 2])
    inds = np.zeros(n)
    files = np.array(files)

    for i in range(n):
        file = files[i]
        meta = open(savedir + file, "rb").readlines(300)
        xy[i, 0] = float(meta[11].decode())
        xy[i, 1] = float(meta[13].decode())
        # xy[i, 0] = float(meta[7].decode())
        # xy[i, 1] = float(meta[9].decode())
        inds[i] = i
        # wl, int[i] = np.loadtxt(open(savedir+file,"rb"),delimiter=",",skiprows=12,unpack=True)


    if switch_xy:
        xy = np.fliplr(xy)

    if flip_x:
        xy[:, 0] = np.max(xy[:, 0]) - xy[:, 0]
    else:
        xy[:, 0] = xy[:, 0] - np.min(xy[:, 0])

    if flip_y:
        xy[:, 1] = np.max(xy[:, 1]) - xy[:, 1]
    else:
        xy[:, 1] = xy[:, 1] - np.min(xy[:, 1])

    # reverse x values, because sample is upside down in microscope
    # ordered = np.argsort(xy[:, 1])
    # xy = xy[ordered, :]
    # xy[:, 0] = xy[::-1, 0]
    # xy[:, 1] = xy[::-1, 1]
    # files = files[ordered]
    # n = xy.shape[0]

    inds, xy, ids = fit_grid(xy, nx, ny)
    files = files[inds]

    # plt.figure(figsize=(5, 5))
    # plt.scatter(xy[:, 0], xy[:, 1], color="red", marker='.')
    # #plt.scatter(grid[:, 0], grid[:, 1], color="blue", marker="x")
    # # plt.show()
    # for x, y, s in zip(xy[:, 0], xy[:, 1], ids):
    #     plt.text(x, y + 1, s)
    #
    # plt.savefig(savedir + "grid.png", dpi=300)
    # plt.close()

    # plt.figure(figsize=(5, 5))
    # plt.scatter(xy[:, 0], xy[:, 1], color="red", marker='.')
    # #plt.scatter(grid[:, 0], grid[:, 1], color="blue", marker="x")
    # # plt.show()
    # for x, y, s in zip(xy[:, 0], xy[:, 1], ids):
    #     plt.text(x, y + 1, s)
    #
    # plt.savefig(savedir + "grid.png", dpi=300)
    # plt.close()

    f = open(savedir + "ids.csv", 'w')
    f.write("x,y,filename,id" + "\r\n")
    for i in range(len(ids)):
        f.write(str(xy[i, 0]) + "," + str(xy[i, 1]) + "," + str(files[i]) + "," + str(ids[i]) + "\r\n")
    f.close()

    return files, ids, xy


def fit_grid(xy, nx, ny):
    # plt.plot(xy[:,0],xy[:,1])
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
                points[i + j * ny, :] = a0 + a * i + b * j
                ids[i + j * ny] = (letters[i] + "{0:d}".format(ny - j))
                # points[i + j * ny, :] = a0 + a * i + b * j
                # ids[i + j * ny] = (letters[nx - i - 1] + "{0:d}".format(j + 1))
                # points[i + j * ny, :] = a0 + a * i + b * j
                # ids[i + j * ny] = (letters[i] + "{0:d}".format(ny - j))
        # ordered = np.argsort(points[:, 0])
        # points = points[ordered, :]
        # points[:, :] = points[::-1, :]
        # ids = ids[ordered]
        return points, ids

    # calculate min dist between sets of points
    def calc_mindists(points1, points2):
        dists = np.zeros(points1.shape[0])
        indices = np.zeros(points1.shape[0], dtype=np.int)
        buf = np.zeros(points2.shape[0])
        weights = np.zeros(points2.shape[0])
        for i in range(points1.shape[0]):
            for j in range(points2.shape[0]):
                # buf[j] = np.sqrt( np.sum( np.square( points2[j,:] - points1[i,:] ) ) )
                buf[j] = np.sqrt(np.sum(np.square(np.subtract(points2[j, :],points1[i, :]))))
            indices[i] = np.argmin(buf)
            weights[indices[i]] += 1;
            dists[i] = buf[indices[i]] * weights[indices[i]]
            #dists[i] = buf[indices[i]]
        return dists, indices

    # function for adding up distances of points between two grids
    def grid_diff(points1, points2):
        return np.sum(calc_mindists(points1, points2))

    rec = minimum_bounding_rectangle(xy)
    l = rec[:, 0] ** 2 + rec[:, 1] ** 2
    ordered = np.argsort(l)
    rec = rec[ordered]
    rec = rec[:-1]
    # print(rec)
    x0, y0 = rec[0, :]
    x_ind = np.argmax(np.dot(rec, [1, 0]))
    y_ind = np.argmax(np.dot(rec, [0, 1]))
    ax, ay = (rec[x_ind, :] - rec[0, :]) / (nx - 1)
    bx, by = (rec[y_ind, :] - rec[0, :]) / (ny - 1)

    # error function for minimizing
    def calc_error(params):
        grid, ids = make_grid(nx, ny, params[0], params[1], params[2], params[3], params[4], params[5])
        # grid, ids = make_grid(nx,ny,x0,y0,params[0],params[1],params[2],params[3])
        return grid_diff(xy, grid)

    start = np.array([x0, y0, ax, ay, bx, by])
    bnds = ((0, 2 * x0), (0, 2 * y0), (-ax, ax * 2), (-ay, ay * 2), (-bx, bx * 2), (-by, by * 2))

    # print(start)
    # grid, ids = make_grid(nx, ny, start[0], start[1], start[2], start[3], start[4], start[5])
    # plt.scatter(xy[:, 0], xy[:, 1], color="red",marker='.')
    # plt.scatter(grid[:, 0], grid[:, 1], color = "blue",marker="x")
    # plt.plot(rec[:, 0], rec[:, 1], "bo")
    # plt.arrow(x0, y0, ax, ay, head_width=0.2, head_length=0.2)
    # plt.arrow(x0, y0, bx, by, head_width=0.2, head_length=0.2)
    # for x, y, s in zip(grid[:, 0], grid[:, 1], ids):
    #     plt.text(x, y + 1, s)
    # plt.savefig(savedir + "grid_start.pdf", format='pdf')
    # plt.close()

    res = minimize(calc_error, start, method='SLSQP', tol=1e-12, options={'disp': True, 'maxiter': 500})
    # res = minimize(calc_error, start, method='SLSQP',bounds=bnds,tol=1e-12, options={ 'disp': True, 'maxiter': 500})
    # res = minimize(calc_error, start, method='L-BFGS-B', jac=False,bounds=bnds, options={'disp': True, 'maxiter': 500})
    # res = minimize(calc_error, start, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
    grid, ids = make_grid(nx, ny, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5])

    d, inds = calc_mindists(grid, xy)
    #print(inds)
    validpoints = np.where(d < np.sqrt(np.sum(np.square(np.subtract(xy[0,:],xy[1,:]))))/2)[0]#np.where(d < 1e6)[0]
    #print(len(validpoints))
    ids = ids[validpoints]
    inds = inds[validpoints]
    grid = grid[validpoints, :]

    xy = xy[inds, :]
    # grid = grid[inds,:]
    # ids = ids[inds]

    plt.scatter(xy[:, 0], xy[:, 1], color="red", marker='.')
    plt.scatter(grid[:, 0], grid[:, 1], color="blue", marker="x")
    # plt.plot(rec[:, 0], rec[:, 1], "bo")
    # plt.arrow(x0, y0, ax, ay, head_width=0.2, head_length=0.2)
    # plt.arrow(x0, y0, bx, by, head_width=0.2, head_length=0.2)
    for x, y, s in zip(xy[:, 0], xy[:, 1], ids):
        plt.text(x, y + 1, s)
    plt.savefig(savedir + "grid_start.pdf", dpi=300)
    plt.close()

    return inds, xy, ids, [ax,ay], [bx,by]


if __name__ == "__main__":

    if len(sys.argv) == 3:
        path = sys.argv[1]
        sample = sys.argv[2]
    else:
        print("No parameters given, using default.")
        # RuntimeError("Too much/less arguments")
        path = '/home/sei/Spektren/2C1/'
        sample = '2C1_75hept_B2'

    # grid dimensions
    nx = 7
    ny = 7
    maxwl = 900
    minwl = 450

    # for sample in samples:

    savedir = path + sample + '/'

    try:
        os.mkdir(savedir + "plots/")
    except:
        pass
    try:
        os.mkdir(savedir + "specs/")
    except:
        pass

    files, ids, grid = fit_grid(savedir, nx, ny)
    for i in range(len(files)):
        print(files[i] + ' ' + ids[i])
