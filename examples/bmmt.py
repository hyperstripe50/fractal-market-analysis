from fma.mmar.timeseries import __simulate_bmmt
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np 

if __name__ == '__main__':
    x1, y1 = np.array(__simulate_bmmt(9, M=[.6, .4], randomize=True))

    plt.plot(x1,y1, 'b-')

    z1 = np.array(y1)
    z2 = np.array([0] * len(y1))

    plt.fill_between(x1, y1, 0,
                 where=(z1 < z2),
                 alpha=0.30, color='red', interpolate=True)
    plt.fill_between(x1, y1, 0,
                 where=(z1 >= z2),
                 alpha=0.30, color='green', interpolate=True)

    plt.show()