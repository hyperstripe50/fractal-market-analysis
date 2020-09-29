import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from fma.mmar.brownian_motion import BrownianMotion

if __name__ == '__main__':
    bm1 = BrownianMotion(12, .457, .603, randomize=False)
    bm2 = BrownianMotion(12, .457, .603, randomize=False)
    x1, y1 = bm1.simulate()
    x2, y2 = bm2.simulate()

    plt.plot(x1,y1, 'b-')
    plt.plot(x2,y2, 'g-')

    plt.show()