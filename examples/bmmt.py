from fma.mmar.timeseries import __simulate_bmmt
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np 

if __name__ == '__main__':
    x1, y1 = np.array(__simulate_bmmt(2, M=[.6, .2, .2], randomize=True))
    print(x1)
    plt.plot(x1,y1, 'b-')

    plt.show()