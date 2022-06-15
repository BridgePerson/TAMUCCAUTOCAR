"""
A simple example of an animated plot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
ax.set(xlim=(0, 10), ylim=(0, 10), xlabel='x axis', ylabel='y axis', title='Planet\'s orbit')
angle= 90
x = np.arange(0, 10, 0.01)        # x-array
line, = ax.plot(x, x)

def animate(i):
    line.set_ydata(x+i)  # update the data
    return line,

#Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
    interval=25, blit=True)
plt.show()