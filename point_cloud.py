import numpy as np
import matplotlib.pyplot as plt


'''
Implementation of Liu et al's Automatic Least Squares Projection method
'''


'''
Given a point p and a projection vector n
Project it onto C by minimizing 
E(p) = sum_i w_i || p* - p_i || ^ 2
'''
def directedProjection(p, n, pts, weights):
    # Ensure given inputs are valid
    np.testing.assert_equal(p.shape, n.shape)
    np.testing.assert_equal(len(pts), len(weights))

    dim = len(p)
    c_0 = weights.sum()
    c = np.array([ (pts[:, d] * weights).sum() for d in range(dim)])
    return (np.dot(c, n) / c_0 - np.dot(p, n))

# Compute the projection direction n
def projDir(pts, weights, p):
    np.testing.assert_equal(len(pts), len(weights))

    dim = len(p)
    c_0 = weights.sum()
    c = np.array([ (pts[:, d] * weights).sum() for d in range(dim)]) 
    m = (c / c_0) - p
    return m / np.linalg.norm(m)

# One step of the LSP method
def projectStep(p, pts):
    '''
    Input:
        p = point we want to project
        pts = points we are projecting onto
    Output:
        n = projection direction
        t = the value such that pnew = p + tn
        weights
    '''
    # Compute weights
    weights = np.array([1 / (1 + np.linalg.norm(p - p_i) ** 4) for p_i in pts])

    # Compute the projection direction
    n = projDir(pts, weights, p)

    # Compute the projection through Directed Projection
    t = directedProjection(p, n, pts, weights)
    return t, n, weights


# rewrote this function to be more inline and efficient
def LSP(p, pts, maxSteps):
    MAX_STEPS = maxSteps
    k = 0
    t = 0

    while k < MAX_STEPS:
        # Compute weights
        weights = np.array([1 / (1 + np.linalg.norm(p - p_i) ** 4) for p_i in pts])

        # Compute projection direction
        dim = len(p)
        c_0 = weights.sum()
        c = np.array([ (pts[:, d] * weights).sum() for d in range(dim)]) 
        m = (c / c_0) - p
        n = m / np.linalg.norm(m)

        # Compute t
        t = (np.dot(c, n) / c_0 - np.dot(p, n))

        # Update the active indices by looking at the weights
        w_max = weights.max()
        w_mean = c_0 / len(weights)
        w_lim = w_mean
        if k < 11:
            w_lim += (w_max - w_mean) / (12 - k)
        else:
            w_lim += (w_max - w_mean) / 2

        pts = np.array([p_i for i, p_i in enumerate(pts) if weights[i] >= w_lim])
        
        if len(pts) == 0:
            break
        k += 1

    return p + t * n

