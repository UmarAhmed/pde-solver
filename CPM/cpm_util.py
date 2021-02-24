import numpy as np
import bisect

'''
Contains many utility functions used in the Closest Point Method
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
def createBand(pts, dist, dx):
    dim = 2
    p = 3
    order = 2
    threshold = dx * 1.0001 * np.sqrt( (dim - 1) * ((p+1)/2)**2 + ((order/2 + (p+1)/2)**2));
    band = [i for i, p in enumerate(pts) if dist(p) <= threshold]
    return band

# Construct laplacian matrix of dimension len(band) x len(band)
# N = total number of points in the grid
# TODO see how to generalize this for higher order (currently order = 2)
def createLaplacian(N, grid_width, band, dx = 0.1, order = 2):
    coefficients = np.array([-4, 1, 1, 1, 1]) / (dx * dx) 
    laplacian = np.zeros( shape = (len(band), N) )

    for i, idx in enumerate(band):
        stencil = [idx, idx - 1, idx + 1, idx - grid_width, idx + grid_width]
        laplacian[i][stencil] = coefficients

    return laplacian[:, band] # Return only columns in the band


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


# Return 1D lagrange weight of x wrt to arr[i] in arr
def lagrange1D(x, arr, i):
    result = 1
    for j, x_j in enumerate(arr):
        if i == j:
            continue
        result *= (x - arr[j]) / (arr[i] - arr[j])
    return result


# Create interpolation matrix using Lagrange interpolation
# We are interpolating pts on the product grid generated by x_pts y_pts
def createInterpMatrix(x_pts, y_pts, pts, band):
    E = []
    for p in pts:    
        # Find the 4 x and y pts closest to p
        n_x = kClosest(x_pts, p[0])
        n_y = kClosest(y_pts, p[1])

        x_stencil = x_pts[n_x]
        y_stencil = y_pts[n_y]

        # Initialize row that we want to put the weights into
        row = np.zeros( shape = len(band) )
        
        # Set the weights using Lagrangian interpolation
        for i in range(len(x_stencil)):
            for j in range(len(y_stencil)):
                w = lagrange1D(p[0], x_stencil, i) * lagrange1D(p[1], y_stencil, j)
                k = len(x_pts) * n_y[j] + n_x[i]
                # k is an index in pts, we must find out the index in the band
                band_k = bisect.bisect_left(band, k)
                row[band_k] = w

        # The Lagrangian weights should add up to 1
        np.testing.assert_almost_equal(row.sum(), 1)

        E.append(row)

    return np.array(E)


# Performs the stabilization from the ICPM paper
def stab(L, E):
    L_diag = np.diag(L) * np.identity(L.shape[0])
    M = L_diag + (L - L_diag) @ E
    return M

