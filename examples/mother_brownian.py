from fma.mmar.timeseries import __compute_fbm
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np 

if __name__ == '__main__':
    x1, y1 = np.array(__compute_fbm(12, .457, .603))
    x2, y2 = np.array(__compute_fbm(12, .131, .603))

    plt.plot(x1,y1, 'b-')
    plt.plot(x2,y2, 'g-')

    plt.show()