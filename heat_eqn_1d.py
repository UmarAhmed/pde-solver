from matplotlib import pyplot as plt
from matplotlib import animation
import math


DELTA_X = 0.5
DELTA_T = 0.05


def gen_range(start, end, step):
    result = []
    while start <= end:
        result.append(start)
        start += step
    return result


'''
[a, b] is the domain of interest (spatially)
[0, stop_time] is the domain of interest temporally
f indicates the initial distribution over [a, b] at t = 0
we assume the endpoints remain constant
'''

def solveEquation(a, b, f, stop_time):
    result = [] # will store the values at each time step so we can animate it later
    
    nodes = gen_range(a, b, DELTA_X)
    initial_vals = [f(x) for x in nodes]
    result.append(initial_vals)    

    cur_time = DELTA_T
    while cur_time <= stop_time:
        # Approximate the distribution at t = cur_time
        new_vals = [initial_vals[0]]
        n = len(nodes)

        # Update all of the nodes (except the first and last)
        # We use Euler time stepping here and a 3 point approximation
        for i in range(1, n - 1):
            new_node = initial_vals[i] + DELTA_T * (initial_vals[i + 1] + initial_vals[i - 1] - 2 * initial_vals[i]) / (DELTA_X * DELTA_X)
            new_vals.append(new_node)

        new_vals.append(initial_vals[-1])
        initial_vals = new_vals[::]
        
        result.append(new_vals)
        cur_time += DELTA_T
        
    return result



f = lambda x: 3 * math.e**(-x*x)
result = solveEquation(-10, 10, f, stop_time = 5)
nodes = gen_range(-10, 10, DELTA_X)


# Set up plot
fig, ax = plt.subplots()
ax.set_ylabel('Temperature')
ax.set_xlabel('Position')

line, = ax.plot(nodes, result[0])

def animate(i):
    line.set_data(nodes, result[i])
    return line,

ani = animation.FuncAnimation(fig, animate, frames = len(result))
#ani.save('animation.mp4')
