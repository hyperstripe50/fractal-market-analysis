from fma.mmar.timeseries import __simulate_bmmt
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np 

if __name__ == '__main__':
    x1, y1 = np.array(__simulate_bmmt(9, M=[.6, .4], randomize=True))

    plt.plot(x1,y1, 'b-')
    plt.show()