from matplotlib import pyplot as plt
import numpy as np

  
# Note that for stability we should have 2 * DELTA_T <= DELTA_X * DELTA_X 
DELTA_X = 0.5
DELTA_T = 0.02
STOP_TIME = 1


def gen_range(start, end, step):
    result = []
    while start <= end:
        result.append(start)
        start += step
    return result
        

# List of x and y nodes
x_pts = gen_range(-1, 1, step = DELTA_X)
y_pts = gen_range(-1, 1, step = DELTA_X)
n = len(x_pts)


def solve(f):
    # initial distribution
    last = [[f(x, y) for x in x_pts] for y in y_pts]
  
    result = [last]

    t = 0
    while t <= STOP_TIME:
        new = np.array(last[:])
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                new[i][j] = last[i + 1][j] + last[i - 1][j] + last[i][j - 1] + last[i][j + 1] - 4 * last[i][j]
                new[i][j] *= DELTA_T / (DELTA_X * DELTA_X)
                new[i][j] += last[i][j]
        
        last = new
        result.append(new)
        t += DELTA_T
    return result
        

def f(x, y):
    return np.exp(-x * x - y * y)
   
a = solve(f)

for i in range(len(a)):
    if i % 10 == 0:
        plt.imshow(a[i], cmap='hot')
        plt.show()
