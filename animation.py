import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:/Users/epont/TAMUC Project/poses.csv')
for i, j in df.iterrows():
    print(j['x'],j['y'],j['angle'])
    #plt.scatter(j['x'],j['y']) to plot dots

    plt.plot(j['x'],j['y'],marker=(3,0,j['angle']), markersize=20, linestyle='None')


x_graph = np.linspace(0, 10, 100)
y_graph = x_graph
trail = 40

fig, ax = plt.subplots()
ax.set(xlim=(0, 10), ylim=(0, 10), xlabel='x axis', ylabel='y axis', title='Planet\'s orbit')


def polar_animator(i, trail=40):
    l1.set_data((x_graph[i - trail:i])*90, (y_graph[i - trail:i]))
    print(l1)
    return l1,


l1, = ax.plot([], [], marker=(3,0,90) , markevery=[-1])
#l2, = ax.plot([0.8], [0], marker='o') This is how you graph still points



ani = animation.FuncAnimation(fig, polar_animator, frames=len(x_graph), fargs=(trail,), interval=5, blit=True)

ani.save('basic_animation.gif', fps=10)

plt.show()