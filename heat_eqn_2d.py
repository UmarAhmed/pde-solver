import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation

  
# Note that for stability we should have 2 * DELTA_T <= DELTA_X * DELTA_X 
DELTA_X = 0.5
DELTA_T = 0.02
STOP_TIME = 1


def gen_range(start, end, step):
    result = []
    while start <= end:
        result.append(start)
        start += step
    return np.array(result)
        

'''
Initialize the grid along with L and G
    all_pts = list of all nodes (as tuples)
    L       = list of indices i such that all_pts[i] is an interpolation point
    G       = list of indices i such that all_pts[i] is a ghost point
'''
DELTA_X = 0.1

# List of x and y nodes
x_pts = gen_range(-1, 1, step = DELTA_X)
y_pts = gen_range(-1, 1, step = DELTA_X)
num_nodes = len(x_pts)

#all_pts = np.transpose([np.tile(x_pts, num_nodes), np.repeat(y_pts, num_nodes)])
all_pts = []
for x in x_pts:
    for y in y_pts:
        p = np.array( (x, y) )
        all_pts.append(p)
all_pts = np.array(all_pts)


L = []
G = []

for idx, p in enumerate(all_pts):
    # Right now we consider p a ghost point if its on the border
    if np.abs(p[0]) > 0.99 or np.abs(p[1]) > 0.99:
        G.append(idx)
    else:
        L.append(idx)

'''
Generate the Laplacian operator 
'''

laplacian = []

for idx in L:
    row = np.zeros(shape = len(all_pts) )

    row[idx] = -4 / (DELTA_X * DELTA_X)
    row[idx - 1] = row[idx + 1] = 1 / (DELTA_X * DELTA_X)
    row[idx - num_nodes] = row[idx + num_nodes] = 1 / (DELTA_X * DELTA_X)
    
    laplacian.append(row)

laplacian = np.array(laplacian)



'''
Use Explicit time stepping to solve the heat equation
'''

DELTA_T = 0.00025
STOP_TIME = 2

def cpmSolve(init):
    # Initialize u to be the initial distribution
    u = np.array( [init(x, y) for (x, y) in all_pts] )
    result = [u]

    t = 0
    while t <= STOP_TIME:
        # Solve u on the grid using forward Euler time stepping
        unew = u.copy()
        right = DELTA_T * (laplacian @ u)
        
        # Update u
        for idx, val in enumerate(L):
            unew[val] += right[idx]

        # Record the solution into an array for plotting later
        result.append(unew)
        
        # Updates
        u = unew
        t += DELTA_T

    return result
  
  
  
  def t(x, y):
    return np.exp(-x * x - y * y)

r = cpmSolve(t)
r = np.array(r)


fig, a = plt.subplots()

def animate(i):
    a.cla()
    grid = r[i].reshape(num_nodes, num_nodes)
    sns.heatmap(grid, vmin = 0, vmax = 1, ax = a, cbar = False)

ani = animation.FuncAnimation(fig, animate, frames = len(r), interval = DELTA_T * 1000)

  
