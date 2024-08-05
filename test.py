
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
from RadarSim import *
from math import sqrt
import numpy as np


def HJacobian_at(x):
    horiz_dist = x[0]
    altitude   = x[2]
    denom = sqrt(horiz_dist**2 + altitude**2)
    return array ([[horiz_dist/denom, 0., altitude/denom]])

def hx(x):
    return (x[0]**2 + x[2]**2) ** 0.5

locations = []
with open("Dataset/lidar.csv", "r") as file:
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        arr = line.split(",")
        locations.append([float(arr[0]), float(arr[1])])
file.close()

dt = 0.05
rk = ExtendedKalmanFilter(dim_x=3, dim_z=1)
radar = RadarSim(dt, pos=0., vel=locations[0][0], alt=locations[0][1])



# make an imperfect starting guess
rk.x = array([radar.pos,locations[0][0],locations[0][1]])


rk.F = eye(3) + array([[0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]])*dt

rk.R = radar.alt * 0.05 # 5% of distance
rk.Q = array([[0, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]) * 0.001
rk.P *= 50

xs = []
track = []
for i in range(len(locations)):
    z = radar.get_range()
    track.append((radar.pos, locations[i][0], locations[i][1]))
    rk.update(array([z]), HJacobian_at, hx)
    xs.append(rk.x)
    rk.predict()

xs = asarray(xs)
track = asarray(track)
a = 0
n = 0
for i in range(len(xs)):
    predict_x = xs[i,1]
    predict_y = xs[i,2]
    track_x = track[i,1]
    track_y = track[i,2]
    data = str(predict_x)+" "+str(predict_y)+" === "+str(track_x)+" "+str(track_y)
    predict = np.cumsum(xs[i])
    original = np.cumsum(track[i])
    variation = np.sum(predict) - np.sum(original)
    if variation < 0:
        print(str(variation)+" No "+data)
        n +=1
    else:
        print(str(variation)+" Alarm "+data)
        a += 1
print(n)
print(a)
    
