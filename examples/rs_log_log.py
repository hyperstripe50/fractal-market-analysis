import matplotlib.pyplot as plt; plt.style.use('ggplot')
from fma.rs.metrics import __get_obv, __to_log_returns_series, __get_ar1_residuals, __compute_Hc
from fma.rs.plots import __log_log_plot
import numpy as np

if __name__ == '__main__':
    # Load from Dollar Yen historical exchange rate
    series = np.genfromtxt('datasets/dollar-yen-exchange-rate-historical-chart.csv', delimiter=',')[::1,1] # this dataset is the best I can find to verify with Peters FMH. Expected values: H=0.642, c=-0.187

    # calculate log returns and AR(1) residuals as per Peters FMH p.62
    obv = __get_obv(series)
    series = __to_log_returns_series(series[:obv])
    series = __get_ar1_residuals(series)

    # Evaluate Hurst equation
    H, c, data = __compute_Hc(series)
    print("H={:.4f}, c={:.4f}".format(H,c)) # random walk should possess brownian motion Hurst statistics e.g. H=0.5

    #Log log plot
    __log_log_plot(data[0],data[1],H,c, V_stat=False)