import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from point_cloud import *
from cpm_util import *


'''
Define the surface: point cloud circle
'''

t = np.linspace(0, np.pi * 2, 500)
x = np.cos(t)
y = np.sin(t)

surface = np.stack( (x, y), 1)

'''
Construct:
    x_pts, y_pts, all_pts, DELTA_X
'''

DELTA_X = 0.1

x_pts = np.arange(-2, 2 + DELTA_X, step = DELTA_X)
y_pts = np.arange(-2, 2 + DELTA_X, step = DELTA_X)
all_pts = np.transpose([np.tile(x_pts, len(x_pts)), np.repeat(y_pts, len(y_pts))])


'''
Create band and reduce closest points to just the band
'''
cp_all_pts = np.array([LSP(q, surface) for q in all_pts])

dim = 2
p = 3
order = 2
threshold = DELTA_X * 1.0001 * np.sqrt( (dim - 1) * ((p+1)/2)**2 + ((order/2 + (p+1)/2)**2))

band = [i for i, p in enumerate(all_pts) if np.linalg.norm(cp_all_pts[i] - p) <= threshold ]

# Only care about closest points in the band
cp_pts = cp_all_pts[band]

'''
Construct the test and solution function
'''

# Desired solution for u
def solnFn(p):
    angle = np.arctan2(p[1], p[0])
    if angle < 0:
        angle += np.pi * 2
    return np.sin(angle) + np.sin(12 * angle)

# The f (or RHS) that is given to us
def f(p):
    angle = np.arctan2(p[1], p[0])
    if angle < 0:
        angle += np.pi * 2
    return - np.sin(angle) - 144 * np.sin(12 * angle)
    
  
'''
Compute matrices and vectors needed in the Jacobi iteration procedure
'''

E = createInterpMatrix(x_pts, y_pts, cp_pts, band)
E = sparse.csr_matrix(E)

A = sparseLaplacian(band, N = len(all_pts), grid_width=len(x_pts), dx=DELTA_X)
b = sparse.csr_matrix([f(all_pts[i]) for i in band]).transpose()
real = sparse.csr_matrix( [solnFn(p) for p in all_pts[band] ] ).transpose()


d = sparse.diags(A.diagonal())
d = sparse.csr_matrix(d)
diagInv = sparse.linalg.inv(d)
woDiag = A - d;
M = E * diagInv 

'''
Do Ruuth-Merriman Jacobi Iteration
'''


# Start with this guess
def init(p):
    angle = np.arctan2(p[1], p[0])
    if angle < 0:
        angle += np.pi * 2
    return np.sin(angle)


start = time.time()

u = sparse.csr_matrix([init(all_pts[i]) for i in band] ).transpose();

k = 0
maxSteps = 10000
delta = 1.
goal = 0.0000001

while k < maxSteps and delta > goal:
    unew = M * (b - woDiag * u)
    delta = sparse.linalg.norm(unew - u)
    u = unew
    k += 1

stop = time.time()


print('Time:', stop - start)
print('dx:', DELTA_X)
print('Steps:', k)

# Parameterize over the circle
theta = np.linspace(0, np.pi * 2, 100)
r = np.ones( len(theta) )

# Convert (radius, theta) to (rcos theta, rsin theta)
def pol2cart(radius, angle):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return (x, y)

# For each theta get a point on the circle
plot_pts = np.array([pol2cart(r[i], theta[i]) for i in range(len(theta))])

# Interpolation for plotting
Eplot = createInterpMatrix(x_pts, y_pts, plot_pts, band)

# Plot our solution and the actual result
circplot = Eplot @ u
circplot = np.squeeze(np.asarray(circplot))
exactplot = np.array([solnFn(p) for p in plot_pts ])

print('Error = ', abs(exactplot - circplot).max() )



fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

ax.set_title('Comparison of actual and expected solutions')
ax.plot(theta, circplot, label="Our answer")
ax.plot(theta, exactplot, label="Exact answer")
plt.legend(loc='best')
plt.show()


# Plot the error over the circle
fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(111)
ax2.set_title('Error over the circle')
ax2.plot(theta, abs(circplot - exactplot))

plt.show()







