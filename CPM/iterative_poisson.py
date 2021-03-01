# -*- coding: utf-8 -*-
"""

**Iterative Solver for the Poisson Equation on the Circle**


We want to solve $\Delta_S u = f = -9 \sin(3 \theta) -4 \cos(2 \theta) $ over the circle. The analytic solution is known to be $u = \sin(2 \theta) + \cos(3 \theta)$.

We construct the Laplacian and Interpolation matrices L and E as usual.
Then we use Ruuth-Merriman Jacobi Iteration, which consists of repeatedly applying two steps:


1.   $u \leftarrow \mathrm{diag}(L)^{-1} (f - (L - \mathrm{diag}(L)) u ) $
2.   $u \leftarrow E u$

This is similar to the standard Jacobi iteration procedure except we apply the closest point extension matrix $E$ after each iteration.
The reason we use an iterative procedure rather than simply inverting the matrix in this case is that the matrix $M = EL$ is actually singular.
The iterative method can also be faster in cases with sparse matrices (which is what we have).


The algorithm is from [Chen and MacDonald](https://arxiv.org/pdf/1307.4354.pdf)
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from cpm_util import *
from scipy import sparse
import time

'''
Construct:
    x_pts, y_pts, all_pts, band, DELTA_X
'''
DELTA_X = 0.1

# x_pts times y_pts is the cartesian product grid we will work over
x_pts = np.arange(-2, 2 + DELTA_X, step = DELTA_X)
y_pts = np.arange(-2, 2 + DELTA_X, step = DELTA_X)
num_nodes = len(x_pts)

all_pts = np.transpose([np.tile(x_pts, num_nodes), np.repeat(y_pts, num_nodes)])

# The band is a list of indices of points in all_pts that we actually compute over
band = createBand(all_pts, dist, DELTA_X)

'''
Construct the test and solution function
'''

# Desired solution for u
def solnFn(p):
    angle = np.arctan2(p[1], p[0])
    if angle < 0:
        angle += np.pi * 2
    return np.sin(3 * angle) + np.cos(2 * angle)

# The f (or RHS) that is given to us
def f(p):
    angle = np.arctan2(p[1], p[0])
    if angle < 0:
        angle += np.pi * 2
    return -9 * np.sin(3 * angle) - 4 * np.cos(2 * angle)

"""Note that $\Delta_s u = \frac{1}{r^2} u_{\theta \theta} + \frac{1}{r} u_r + u_{rr}$ 
this tells us how to take the Laplacian of a function in terms of $r$ and $\theta$."""

'''
Compute matrices and vectors needed in the Jacobi iteration procedure
'''
'''
Compute matrices and vectors needed in the Jacobi iteration procedure
'''

# TODO rewrite the interpolation matrix fn to return sparse matrix 
cp_all_pts = np.array([cp(all_pts[i]) for i in band])
E = createInterpMatrix(x_pts, y_pts, cp_all_pts, band)
E = sparse.csr_matrix(E)

A = sparseLaplacian(band, N = len(all_pts), grid_width=num_nodes, dx=DELTA_X)
b = sparse.csr_matrix([f(all_pts[i]) for i in band]).transpose()
real = sparse.csr_matrix( [solnFn(p) for p in all_pts[band] ] ).transpose()


# diagonal inverse
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
    return np.sin(3 * angle)


start = time.time()

u = sparse.csr_matrix([init(all_pts[i]) for i in band] ).transpose();

k = 0
maxSteps = 10000
delta = 1.
goal = 0.000001

while k < maxSteps and delta > goal:
    unew = M * (b - woDiag * u)
    delta = sparse.linalg.norm(unew - u)
    u = unew
    k += 1

stop = time.time()


print('Time:', stop - start)
print('dx:', DELTA_X)
print('Steps:', k)

# Account for translation
u = u.toarray()
u += (real - u).mean()

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
