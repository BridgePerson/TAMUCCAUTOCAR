import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


x_graph = np.linspace(0, 10, 100)
y_graph = x_graph
trail = 60

fig, ax = plt.subplots()
ax.set(xlim=(0, 10), ylim=(0, 10), xlabel='x axis', ylabel='y axis', title='Planet\'s orbit')


def polar_animator(i,trail=60): 
    l1.set_data(x_graph[i - trail:i], y_graph[i - trail:i])
    return l1,


l1, = ax.plot([0,0], [10,10], marker=(3,0,70), markevery=[-1])
#l2, = ax.plot([-0.8], [0], marker='o')

ani = animation.FuncAnimation(fig, polar_animator, frames=len(x_graph), fargs=(trail,), interval=5, blit=True)
ani.save('basic_animation.gif', fps=10)

plt.show()