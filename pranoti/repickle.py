import os
import re

import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

path = '/home/sei/Spektren/pranoti/'

# repickle maps

files = []
for file in os.listdir(path):
    if len(re.findall(r"(map.pkl)", file)) > 0:
        files.append(file)

for i in range(len(files)):
    sample = files[i]
    with open(path + sample, 'rb') as fp:
        x, y, img = pickle.load(fp)
    with open(path + sample[:-4] + '_p2.pkl', 'wb') as fp:
        pickle.dump((x,y,img), fp, protocol=2)

# repickle lines

files = []
for file in os.listdir(path):
    if len(re.findall(r"(line.pkl)", file)) > 0:
        files.append(file)

for i in range(len(files)):
    sample = files[i]
    with open(path + sample, 'rb') as fp:
        x, transmittance = pickle.load(fp)
    with open(path + sample[:-4] + '_p2.pkl', 'wb') as fp:
        pickle.dump((x, transmittance), fp)

# repickle roi points

files = []
for file in os.listdir(path):
    if len(re.findall(r"(roi_points.pkl)", file)) > 0:
        files.append(file)

for i in range(len(files)):
    sample = files[i]
    with open(path + sample, 'rb') as fp:
        points = pickle.load(fp)
    with open(path + sample[:-4] + '_p2.pkl', 'wb') as fp:
        pickle.dump(points, fp)