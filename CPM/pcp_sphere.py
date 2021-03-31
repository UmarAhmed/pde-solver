import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import sparse
import scipy.sparse.linalg

from point_cloud import *
from cpm_util import *

'''
Define the surface: point cloud sphere
''' 
N_phi = 20
phi = np.linspace(0, np.pi, N_phi)
theta = np.linspace(0, 2 * np.pi, 2 * N_phi)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

x = x.reshape(2 * N_phi * N_phi,)
y = y.reshape(2 * N_phi * N_phi,)
z = z.reshape(2 * N_phi * N_phi,)

surface = np.stack( (x, y, z), 1)

'''
Construct:
    Cartesian grid containing surface of interest
'''

DELTA_X = 0.1
bound = 1.2
x_pts = np.arange(-bound, bound + DELTA_X, step = DELTA_X)
y_pts = x_pts 
z_pts = x_pts

# Note that it is okay that x_pts, y_pts, z_pts all point to the same object,
# as we never mutate any of them

N = len(x_pts) * len(y_pts) * len(z_pts)
print(f'Grid contains {N} points')


'''
Define constants
'''
dim = 3
p0 = 3
order = 2
threshold = DELTA_X * 1.00001 * np.sqrt( (dim - 1) * ((p0+1)/2)**2 + ((order/2 + (p0+1)/2)**2))


# for each grid node we want to find a subset of the surface
# which is within some threshold distance
'''
For every point in the cloud, find the closest point in the grid to it
Then walk outwards so that all grid points that are in distance threshold to 
the surface point are considered and we add the point to grid_dict
for each grid node that 
'''
def warm_start(pts, surface, threshold):
    n = len(pts)
    grid_dict = [[] for _ in range(n * n * n)]
    dx = pts[1] - pts[0]
    r = int(np.ceil(threshold / DELTA_X))

    for i, s in enumerate(surface):
        # Find closest grid node to s
        grid_x = round((s[0] - pts[0]) / dx)
        grid_y = round((s[1] - pts[0]) / dx)
        grid_z = round((s[2] - pts[0]) / dx)

        # Update all nodes in the r x r subgrid
        leftx = max(0, grid_x - r)
        rightx = min(n - 1, grid_x + r)

        lefty = max(0, grid_y - r)
        righty = min(n - 1, grid_y + r)

        leftz = max(0, grid_z - r)
        rightz = min(n - 1, grid_z + r)

        for row in range(leftx, rightx + 1):
            for col in range(lefty, righty + 1):
                for za in range(leftz, rightz + 1):
                    grid_cur = [pts[row], pts[col], pts[za]]
                    if np.linalg.norm(s - grid_cur) <= threshold:
                        k = row + n * (col + n * za)
                        grid_dict[k].append(i)
    return grid_dict
  
  
# Compute closest points using LSP + warm start
start = time.time()
starters = warm_start(x_pts, surface, threshold)

# Call LSP and obtain cp_pts and band
n = len(x_pts)
cp_pts = []
band = []
l = 0
for i in range(n):
    for j in range(n):
        for k in range(n):
            if starters[l]:
                band.append(l)
                q = np.array([x_pts[i], x_pts[j], x_pts[k]])
                cp_pts.append( LSP(q, surface[starters[l]], maxSteps = 5) )
            l += 1


stop = time.time()
print(f'warm start + LSP took {stop - start}')
max_len = max([len(a) for a in starters])
print(f'Largest starter has size {max_len}')
print(f'Total number of points is {len(surface)}')

# TODO: setting up the system and solving it
