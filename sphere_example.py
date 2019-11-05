#https://stackoverflow.com/questions/32424670/python-matplotlib-drawing-3d-sphere-with-circumferences

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style('darkgrid')

def intersection_exists(pt1, pt2, ptc, r):
    a = (pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2 + (pt2[2] - pt1[2]) ** 2
    
    b = -2 * ((pt2[0] - pt1[0]) * (ptc[0] - pt1[0]) + (pt2[1] - pt1[1]) * (ptc[1] - pt1[1]) + (pt2[2] - pt1[2]) * (ptc[2] - pt1[2]))

    c = (ptc[0] - pt1[0]) ** 2 + (ptc[1] - pt1[1]) ** 2 + (ptc[2] - pt1[2]) ** 2 - r ** 2

    return (b ** 2 - a * c) > 0 

def intersection_points(pt1, pt2, ptc, r):
    a = (pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2 + (pt2[2] - pt1[2]) ** 2
    b = -2 * ((pt2[0] - pt1[0]) * (ptc[0] - pt1[0]) + (pt2[1] - pt1[1]) * (ptc[1] - pt1[1]) + (pt2[2] - pt1[2]) * (ptc[2] - pt1[2]))
    c = (ptc[0] - pt1[0]) ** 2 + (ptc[1] - pt1[1]) ** 2 + (ptc[2] - pt1[2]) ** 2 - r ** 2

    t1 = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    t2 = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    return np.array([pt1 + t1 * (pt2 - pt1), pt1 + t2 * (pt2 - pt1)])
# fig = plt.figure(figsize=(12,12), dpi=300)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
# ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='g', linewidth=0, alpha=0.5)

line_x = np.array([[0, 0, 0], [1.5, 0, 0]])
ax.plot(line_x[:, 0], line_x[:, 1], line_x[:, 2], c='r')
print (intersection_exists(line_x[0], line_x[1], line_x[0], 1))
inter_points_1 = intersection_points(line_x[0], line_x[1], line_x[0], 1)
print (inter_points_1)
ax.scatter3D(inter_points_1[:, 0], inter_points_1[:, 1], inter_points_1[:, 2], c='r')

line_y = np.array([[0, 0, 0], [0, 1.5, 0]])
ax.plot(line_y[:, 0], line_y[:, 1], line_y[:, 2], c='b')
print (intersection_exists(line_y[0], line_y[1], line_y[0], 1))
inter_points_2 = intersection_points(line_y[0], line_y[1], line_y[0], 1)
print (inter_points_2)
ax.scatter3D(inter_points_2[:, 0], inter_points_2[:, 1], inter_points_2[:, 2], c='b')

t = np.linspace(0, 1, 100)
phi = np.pi / 2
theta = t * phi 

arc = (np.expand_dims((np.sin((1 - t) * phi) / np.sin(phi)), axis=0).T) * np.expand_dims(inter_points_1[0], axis=0) + (np.expand_dims((np.sin(t * phi) / np.sin(phi)), axis=0).T) * np.expand_dims(inter_points_2[0], axis=0)

# print (arc.shape)
# print (inter_points[0])
# print (inter_points.shape)
# print (arc)
ax.plot(arc[:, 0], arc[:, 1], arc[:, 2])
ax.axis('off')

plt.show()
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.0001)
