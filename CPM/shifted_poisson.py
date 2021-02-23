import numpy as np
import bisect
import seaborn as sns
from matplotlib import pyplot as plt


# Constants
testing = False
DELTA_X = 0.1


'''
Helper functions
    cp, dist, createBand, createLaplacian
'''

# Return closest point to x on the circle
def cp(p):
    if np.linalg.norm(p) < 0.00001:
        return (1, 0)
    else:
        return p / np.linalg.norm(p)


# Return distance to the surface, which is a circle in this case
def dist(p):
    return np.linalg.norm(p - cp(p))


# Compute a band: indices of p in pts st dist(p) is at most some computed threshold
# dist is assumed to be a function which returns distance to the surface
def createBand(pts, dist):
    dim = 2
    p = 3
    order = 2
    threshold = DELTA_X * 1.0001 * np.sqrt( (dim - 1) * ((p+1)/2)**2 + ((order/2 + (p+1)/2)**2));
    band = [i for i, p in enumerate(pts) if dist(p) <= threshold]
    return band

# Construct laplacian matrix
# N = total number of points in the grid
# TODO see how to generalize this for higher order (currently order = 2)
def createLaplacian(N, grid_width, band, order = 2):
    coefficients = np.array([-4, 1, 1, 1, 1]) / (DELTA_X * DELTA_X) 
    laplacian = np.zeros( shape = (len(band), N) )

    for i, idx in enumerate(band):
        stencil = [idx, idx - 1, idx + 1, idx - grid_width, idx + grid_width]
        laplacian[i][stencil] = coefficients

    return laplacian[:, band] # Return only columns in the band


'''
Construct:
    x_pts, y_pts, all_pts
    L, G (indices)
'''

x_pts = np.arange(-2, 2 + DELTA_X, step = DELTA_X)
y_pts = np.arange(-2, 2 + DELTA_X, step = DELTA_X)
num_nodes = len(x_pts)
all_pts = np.transpose([np.tile(x_pts, num_nodes), np.repeat(y_pts, num_nodes)])

band = createBand(all_pts, dist)




'''
Construct right hand side
    f = 2 sin(theta) + 145 sin(12 theta)
'''

# Test function is sin(theta) + sin(12 theta) on the circle
def solnFn(p):
    angle = np.arctan2(p[1], p[0])
    if angle < 0:
        angle += np.pi * 2
    return np.sin(angle) + np.sin(12 * angle)

# Solution to shifted poisson for the above fn is u = 2 sin(theta) + 145 sin(theta)
def testFn(p):
    angle = np.arctan2(p[1], p[0])
    if angle < 0:
        angle += np.pi * 2
    return 2 * np.sin(angle) + 145 * np.sin(12 * angle)
    
# Compute discrete closest point extension of f onto the band
f = np.array([testFn(all_pts[i]) for i in band])


'''
Construct:
    E (extension matrix)
'''
# Given a sorted array arr and a scalar val, find the K closest values to val
def kClosest(arr, val, K = 4):
    close = []
    left = bisect.bisect_left(arr, val)
    right = left + 1

    if right >= len(arr):
        left -= 1
        right -= 1

    while len(close) < K:
        if left < 0:
            close.append(right)
            right += 1
            continue
        if right >= len(arr):
            close.append(left)
            left -= 1
            continue

        if abs(val - arr[left]) <= abs(val - arr[right]):
            close.append(left)
            left -= 1
        else:
            close.append(right)
            right += 1
    return close

# Given a point p, return the indices of the K closest x and y pts (considered separately)
def neighbor(p, K = 4):
    close_x = kClosest(x_pts, p[0], K)
    close_y = kClosest(y_pts, p[1], K)
    return close_x, close_y


# Return 1D lagrange weight of x wrt to arr[i] in arr
def lagrange1D(x, arr, i):
    result = 1
    for j, x_j in enumerate(arr):
        if i == j:
            continue
        result *= (x - arr[j]) / (arr[i] - arr[j])
    return result


E = []
for p in all_pts:    
    # Find the closest point to p on the surface
    cp_p = cp(p)

    # Find the x and y pts closest to cp_p
    n_x, n_y = neighbor(cp_p) # TODO make sure neighbor -> kClosest still works with the band

    x_stencil = x_pts[n_x]
    y_stencil = y_pts[n_y]

    # Initialize row that we want to put the weights into
    row = np.zeros( shape = len(band) )
    
    # Set the weights using Lagrangian interpolation
    for i in range(len(x_stencil)):
        for j in range(len(y_stencil)):
            w = lagrange1D(cp_p[0], x_stencil, i) * lagrange1D(cp_p[1], y_stencil, j)
            k = num_nodes * n_y[j] + n_x[i]
            # k is an index in all_pts, we must find out the index in L (we call that L_k)
            band_k = bisect.bisect_left(band, k)
            row[band_k] = w

    # The Lagrangian weights should add up to 1
    if testing:
        np.testing.assert_almost_equal(row.sum(), 1)

    E.append(row)

E = np.array(E)
E[band, :] # trim to only keep points in the band

'''
Solve the system
'''
# Construct M sharp from the ICPM paper ( for increased stability )
def stab(L, E):
    L_diag = np.diag(L) * np.identity(L.shape[0])
    M = L_diag + (L - L_diag) @ E
    return M

laplacian = createLaplacian(N = len(all_pts), grid_width = num_nodes, band = band)
M = stab(laplacian, E)
A = np.identity(M.shape[0]) - M
u = np.linalg.solve(A, f)


# Compute errors
soln = np.array([solnFn(all_pts[i]) for i in band])
error = abs(u - soln)
print('Average error', error.sum() / len(error))
print('Max error', error.max())
