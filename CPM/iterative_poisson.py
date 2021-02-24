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
rhs = np.array([f(all_pts[i]) for i in band])
laplacian = createLaplacian(N = len(all_pts), grid_width = num_nodes, band = band)

cp_all_pts = np.array([cp(all_pts[i]) for i in band])

E = createInterpMatrix(x_pts, y_pts, cp_all_pts, band)

real = np.array( [solnFn(p) for p in all_pts[band] ] ) # solution we want to see


A = laplacian
diagInv = np.identity(A.shape[0]) * (1. / np.diag(A))
woDiag = A - (np.identity(A.shape[0]) * np.diag(A))

'''
Do Ruuth-Merriman Jacobi Iteration
'''

# We start with this guess
def init(p):
    angle = np.arctan2(p[1], p[0])
    if angle < 0:
        angle += np.pi * 2
    return np.sin(angle)

# start w a random guess in [-1, 1] could also use init() here
u = np.random.uniform(low = -1, high = 1, size = len(band))

k = 0
maxSteps = 10000
error = 1.
goal = 0.0000001

while k < maxSteps and error > goal:
    unew = diagInv @ (rhs - woDiag @ u)
    unew = E @ unew
    error = np.linalg.norm(unew - u)

    if k % (maxSteps // 10) == 0:
        print(f"Max Error at step {k}:", abs(unew - real).max())

    u = unew
    k += 1
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
exactplot = np.array([solnFn(p) for p in plot_pts ])

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
