import numpy as np
import bisect
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation

'''
Helper functions
'''

# Return closest point to p on the circle
def cp(p):
    if np.linalg.norm(p) < 0.00001:
        return (1, 0)
    else:
        return p / np.linalg.norm(p)

    
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



'''
Construct: 
    x_pts, y_pts, all_pts
    L, G (indices)
    DELTA_X
'''
DELTA_X = 0.1

x_pts = np.arange(-2, 2 + DELTA_X, step = DELTA_X)
y_pts = np.arange(-2, 2 + DELTA_X, step = DELTA_X)
num_nodes = len(x_pts)
all_pts = np.transpose([np.tile(x_pts, num_nodes), np.repeat(y_pts, num_nodes)])

L = []
G = []

for idx, p in enumerate(all_pts):
    # We consider p a ghost point if its on the border
    if np.abs(p[0]) > 1.99 or np.abs(p[1]) > 1.99:
        G.append(idx)
    else:
        L.append(idx)


'''
Construct:
    laplacian (laplacian operator)
'''

laplacian = []

for idx in L:
    row = np.zeros(shape = len(all_pts) )

    row[idx] = -4 / (DELTA_X * DELTA_X)
    row[idx - 1] = row[idx + 1] = 1 / (DELTA_X * DELTA_X)
    row[idx - num_nodes] = row[idx + num_nodes] = 1 / (DELTA_X * DELTA_X)

    # Make sure the left and right values are actually to the left and right
    for i in range(-1, 1):
        np.testing.assert_almost_equal(all_pts[idx + i][0] + DELTA_X, all_pts[idx + i + 1][0])
        np.testing.assert_almost_equal(all_pts[idx + i][1], all_pts[idx + i + 1][1])

    # Similarly check the top and bottom
    np.testing.assert_almost_equal(all_pts[idx - num_nodes][0], all_pts[idx][0])
    np.testing.assert_almost_equal(all_pts[idx + num_nodes][0], all_pts[idx][0])
    np.testing.assert_almost_equal(all_pts[idx - num_nodes][1] + DELTA_X, all_pts[idx][1])
    np.testing.assert_almost_equal(all_pts[idx + num_nodes][1] - DELTA_X, all_pts[idx][1])

    laplacian.append(row)

laplacian = np.array(laplacian)




'''
Construct:
    E (extension matrix)
'''

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
    n_x, n_y = neighbor(cp_p)

    x_stencil = x_pts[n_x]
    y_stencil = y_pts[n_y]

    # Initialize row that we want to put the weights into
    row = np.zeros( shape = len(L) )
    
    # Set the weights using Lagrangian interpolation
    for i in range(len(x_stencil)):
        for j in range(len(y_stencil)):
            w = lagrange1D(cp_p[0], x_stencil, i) * lagrange1D(cp_p[1], y_stencil, j)
            k = num_nodes * n_y[j] + n_x[i]
            # k is an index in all_pts, we must find out the index in L (we call that L_k)
            L_k = bisect.bisect_left(L, k)
            np.testing.assert_almost_equal(k, L[L_k])
            row[L_k] = w

    # The Lagrangian weights should add up to 1
    np.testing.assert_almost_equal(row.sum(), 1)

    E.append(row)

E = np.array(E)


'''
actual solver here

'''
DELTA_T = 0.002
STOP_TIME = 1

# u is the initial distribution over the domain of interest
def cpmSolve(u):

    result = [u]

    t = 0
    while t <= STOP_TIME:
        # Solve u on the grid using forward Euler time stepping
        unew = u.copy()
        right = DELTA_T * (laplacian @ u)
        
        # Update u
        for idx, val in enumerate(L):
            unew[val] += right[idx]
        
        unew = E @ unew

        # Record the solution into an array for plotting later
        result.append(unew)
        
        # Updates
        u = unew
        t += DELTA_T

    return result

'''
Testing
    Input: cos(theta)
    Analytic solution: exp(-t) cos(theta)
'''

# Test function
def sample(p):
    angle = np.arctan2(p[1], p[0])
    return np.cos(angle)

# Solve the DE
u = np.array([sample(p) for p in all_pts])
r = cpmSolve(u)

# Mask for showing band
ddd = np.zeros(shape = (num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        p = np.array([x_pts[i], y_pts[j]])
        cp_p = cp(p)
        if np.linalg.norm(p - cp_p)<= 0.3606:
            ddd[j][i] = False
        else:
            ddd[j][i] = True
 
# This is for printing out images
for i, a in enumerate(r):
    if i % 100 == 0:
        print('Time:', i * DELTA_T)
        m = a.reshape(len(x_pts), len(y_pts))
        plt.subplots(figsize=(10,8))
        sns.heatmap(m, mask = ddd)
        plt.show()

# Use the following for creating animations
'''
fig, a = plt.subplots()

def animate(i):
    a.cla()
    grid = r[i].reshape(num_nodes, num_nodes)
    sns.heatmap(grid, mask = ddd, vmin = -1, vmax = 1, ax = a, cbar = False)
ani = animation.FuncAnimation(fig, animate, frames = len(r), interval = DELTA_T * 1000)
ani.save("animation.mp4")
'''
