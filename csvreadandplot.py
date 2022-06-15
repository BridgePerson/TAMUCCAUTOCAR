import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as an

def main():


    df = pd.read_csv(r'C:/Users/epont/TAMUC Project/poses.csv')
    for i, j in df.iterrows():
        print(j['x'],j['y'],j['angle'])
        #plt.scatter(j['x'],j['y'])
        plt.plot(j['x'],j['y'],marker=(3,0,j['angle']), markersize=20, linestyle='None')

    plt.imshow
    plt.show()
main()