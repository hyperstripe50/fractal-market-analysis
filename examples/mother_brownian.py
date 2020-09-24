from fma.mmar.timeseries import __compute_fbm
from fma.rs.metrics import __get_obv, __to_log_returns_series, __get_ar1_residuals, __compute_Hc
from fma.rs.plots import __log_log_plot
import numpy as np

if __name__ == '__main__':
    x1, y1 = np.array(__compute_fbm(12, .457, .503))
    x2, y2 = np.array(__compute_fbm(3, .131, .603))

    # plt.plot(x1,y1, 'b-')
    # plt.plot(x2,y2, 'g-')

    # plt.show()

    # calculate log returns and AR(1) residuals as per Peters FMH p.62
    series = y1[1:]
    obv = __get_obv(series)
    series = __to_log_returns_series(series[:obv])
    series = __get_ar1_residuals(series)

    # Evaluate Hurst equation
    H, c, data = __compute_Hc(series)
    print("H={:.4f}, c={:.4f}".format(H,c)) # random walk should possess brownian motion Hurst statistics e.g. H=0.5

    #Log log plot
    __log_log_plot(data[0],data[1],H,c, V_stat=True)