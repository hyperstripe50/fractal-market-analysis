from fma.mmar.timeseries import __compute_fbm
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np 

if __name__ == '__main__':
    fbm1 = np.array(__compute_fbm(10, 4/9, 2/3))
    x = fbm1[:,0]
    y = fbm1[:,1]

    fbm2 = np.array(__compute_fbm(10, 1/9, 2/3))
    y2 = fbm2[:,1]
    x2 = fbm2[:,0]

    print(x)
    plt.plot(x,y, 'b-')
    plt.plot(x2,y2, 'g-')

    plt.show()