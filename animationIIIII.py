import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import pandas as pd

fig, ax = plt.subplots()



def animate(i):
    ## plot scatter point
    sc_1.set_offsets([x_start, y_start])

    ## plot line
    line.set_data(x[:i], y[:i])

    ## plot scatter point
    if i == len(x):
       sc_2.set_offsets([x_end, y_end])

    return sc_1, line, sc_2




df = pd.read_csv(r'C:/Users/epont/TAMUC Project/poses.csv')
for i, j in df.iterrows():
    x_start, y_start = (j['x'], j['y'])
    x_1, y_1 = j['x'], j['y']
    angle=j['angle']
    x_2=x_1*math.degrees(math.cos(math.radians(angle)))
    y_2=y_1*math.degrees(math.cos(math.radians(angle)))
    x_end, y_end = (x_2, y_2)
    x = np.linspace(x_1, x_2, 50)
    y = np.linspace(y_1, y_2, 50)
    sc_1 = ax.scatter([], [], color="green", zorder=4)
    line, = ax.plot([], [], color="crimson", zorder=4)
    sc_2 = ax.scatter([], [], color="gold", zorder=4)
    ani = animation.FuncAnimation(
    fig=fig, func=animate, interval=100, blit=True, save_count=50)
    






#print(x_2, y_2)

plt.xlim((0, 100))
plt.ylim((0,100))







#ani = animation.FuncAnimation(
    #fig=fig, func=animate, interval=100, blit=True, save_count=50)

plt.show()