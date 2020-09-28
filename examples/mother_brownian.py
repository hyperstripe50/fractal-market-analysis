from fma.mmar.timeseries import simulate_bm
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')

if __name__ == '__main__':
    x1, y1 = np.array(simulate_bm(12, .457, .603, randomize=False))
    x2, y2 = np.array(simulate_bm(3, .131, .603, randomize=False))

    plt.plot(x1,y1, 'b-')
    plt.plot(x2,y2, 'g-')

    plt.show()