import numpy as np

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
 def lagrangeND(p, ndgrid, indices):
    result = 1
    n = len(p)
    for i, p_i in enumerate(p):
        result *= lagrange1D(p_i, ndGrid[i], indices[i])
    return result


'''
Test and visualize in 2D
'''
# We have a grid of points
x = [i for i in range(-10, 11)]
y = [i for i in range(-10, 11)]
pts = []
for i in x:
    for j in y:
        pts.append( (i, j) )


# And we know the value of some function f on those points
def f(x, y):
    return x * np.sin(y)

z = [f(i, j) for (i, j) in pts]

# We want to know the value at some more points on the plane
interp_pts = [np.random.uniform(low = -10, high=10, size=(2,)) for _ in range(50)]

# So we will use Lagrange interpolation
z_interp = []


for p in interp_pts:
    result = 0
    val_count = 0
    for a in range(len(x)):
        for b in range(len(y)):
            result += z[val_count] * lagrangeND(p, (x, y), (a, b))
            val_count += 1
    z_interp.append(result)
    
    
total = 0
for i, (x, y) in enumerate(interp_pts):
    total += np.abs(f(x, y) - z_interp[i])
print(total)
ax = plt.axes(projection='3d')

# Plot known data
x_data = [a[0] for a in pts]
y_data = [a[1] for a in pts]
z_data = [a for a in z]
ax.scatter3D(x_data, y_data, z_data)

# Plot interpolated data
x_interp = [a[0] for a in interp_pts]
y_interp = [a[1] for a in interp_pts]
ax.scatter3D(x_interp, y_interp, z_interp)
