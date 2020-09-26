from fma.mmar.timeseries import __compute_fbm
from fma.rs.metrics import __get_obv, __to_log_returns_series, __get_ar1_residuals, __compute_Hc
from fma.rs.plots import __log_log_plot
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')

if __name__ == '__main__':
    x1, y1 = np.array(__compute_fbm(12, .457, .603, randomize=False))
    x2, y2 = np.array(__compute_fbm(3, .131, .603, randomize=False))

    plt.plot(x1,y1, 'b-')
    plt.plot(x2,y2, 'g-')

    plt.show()