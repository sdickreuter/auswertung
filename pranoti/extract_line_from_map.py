import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg



path = '/home/sei/Spektren/pranoti/'

#samples = ['E 10283 A1 1s map']
samples = ['E 10283 E9 0.0s map','E 10283 C1 0.2s map','E 10283 A1 1s map',]

dirs = ['y','x','x',]

i = 0

sample = samples[i]
dir = dirs[i]

with open(path+sample+'.pkl', 'rb') as fp:
    x,y, D = pickle.load(fp)

app = QtGui.QApplication([])

## Create window with two ImageView widgets
win = QtGui.QMainWindow()
win.resize(800, 800)
win.setWindowTitle('pyqtgraph example: DataSlicing')
cw = QtGui.QWidget()
win.setCentralWidget(cw)
l = QtGui.QGridLayout()
cw.setLayout(l)
imv1 = pg.ImageView()
p = pg.plot()
curve = p.plot()
l.addWidget(imv1, 0, 0)
l.addWidget(p, 1, 0)
win.show()

if dir == 'x':
    roi = pg.LineSegmentROI([[2, 50], [x.max()-2, 50]], pen='r')
elif dir == 'y':
    roi = pg.LineSegmentROI([[50, y.max()-2], [50, 2]], pen='r')

imv1.addItem(roi)


def update():
    global D, imv1, curve
    d2 = roi.getArrayRegion(D, imv1.imageItem)
    curve.setData(d2)

roi.sigRegionChanged.connect(update)

## Display the data
imv1.setImage(D)
#imv1.setHistogramRange(-0.01, 0.01)
#imv1.setLevels(-0.003, 0.003)

update()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

transmittance = roi.getArrayRegion(D, imv1.imageItem)

hh = roi.getHandles()
points = [roi.mapToItem(imv1.imageItem, h.pos()) for h in hh]
points = [points[0].x(),points[0].y(),points[1].x(),points[1].y()]

app.quit()
win = None
app = None

print(points)
print(transmittance.shape)
print(D.shape)
print(x.shape)
print(y.shape)

if dir == 'x':
    x = x[0:len(transmittance)]
elif dir == 'y':
    x = y[0:len(transmittance)]

transmittance = transmittance - transmittance[0] + 1


plt.plot(x,transmittance)
plt.show()


with open(path + sample[:-3] + 'line.pkl', 'wb') as fp:
    pickle.dump((x, transmittance), fp)

with open(path + sample[:-3] + 'roi_points.pkl', 'wb') as fp:
    pickle.dump(points, fp)

