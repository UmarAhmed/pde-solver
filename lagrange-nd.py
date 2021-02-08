# Standard 1D lagrange polynomial basis for index i
def lagrange1D(x, arr, i):
    result = 1
    for j, x_j in enumerate(arr):
        if i == j:
            continue
        result *= (x - arr[j]) / (arr[i] - arr[j])
    return result
 
 
# Fetch lagrange weighs for a point p with respect to some point (x[i], y[j]) 
# on the product grid given by the product of x and y
 def lagrange2D(p, ndgrid, indices):
    result = 1
    n = len(p)
    for i, val in enumerate(p):
        result *= lagrange1D(p[i], ndGrid[i], indices[i])
    return result
# Test and visualize in 2d
