import numpy as np
import math
import quandl
from hurst import random_walk
import matplotlib.pyplot as plt; plt.style.use('ggplot')
quandl.ApiConfig.api_key="G1MYyTPSFZkRfPM6MWUp" # for pulling data from quandl

def __to_log_returns_series(x):
    """
    :param x: 1D array of numbers
    :return: 1D array of log returns
    """
    log_returns = np.log(x[1:]/x[:-1])

    return log_returns

def __get_obv(x):
    """
    :param x: 1D array
    :return: number of observations to the lower 100 + 2. Since AR(1) residual is passed on to R/S, we lose two points along the way.
             Thus, we need to start with two more observations than we wish to pass to the R/S section.
    """
    obv = math.floor((len(x)-1)/100)*100 + 2

    return obv

def __get_ar1_residuals(x):
    """
    :param x: 1D array of numbers
    :return: 1D array of AR(1) residuals
    """
    # calculate AR(1) residuals to remove autocorrelation and make data stationary as per Peters FMH p.281
    yi   = x[1:]
    xi   = x[:-1]
    xi2  = np.power(xi, 2)
    ybar = np.mean(yi)
    xbar = np.mean(xi)
    xy   = np.multiply(yi,xi)
    sxx  = len(x) * np.sum(xi2) - np.power(np.sum(xi), 2)
    sxy  = len(x) * np.sum(xy) - np.sum(xi) * np.sum(yi)
    slope = sxy / sxx
    const = ybar - slope*xbar

    ar1_residuals = x[1:] - (const + slope * x[:-1])

    return ar1_residuals

def __get_rs(x):
    """
    :param x: 1D array of numbers
    :return: return R/S for given array
    """
    mean_x = np.sum(x) / len(x)
    rescaled_x = x - mean_x
    Z = np.cumsum(rescaled_x)
    R = max(Z) - min(Z)
    S = np.std(x, ddof=0) # Peters FMH p.62 uses population standard deviation i.e. ddof = 0. Sample standard deviation ddof = 1.

    if R == 0 or S == 0:
        return 0

    return R / S

def __compute_Hc(x):
    """
    :param x: 1D array of numbers
    :return: number representing R/S rescaled range
    """
    i = 0 
    obv = len(x)
    RS = []
    N  = []
    while i < obv - 1:
        i += 1
        n   = math.floor(obv/i)
        num = obv/i
        rs = []
        if n >= num and n > 9: # small values of n produce unstable estimates when sample size is small
            for start in range(0, len(x), n):
                rs.append(__get_rs(x[start:start + n]))
            RS.append(np.mean(rs))
            N.append(n)

    A = np.vstack([np.log10(N), np.ones(len(N))]).T # y = Ap, where A = [[x 1]] and p = [[m], [c]]
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0] # slope (Hurst exponent), intercept (constant); WRT Peters FMH p. 56 eq 4.7 (R/S)_n = c*n^H

    return H, c, [N, RS]

def __log_log_plot(x,y,H,c,show=False):

    """
    :param x: 1D array non log scaled
    :param y: 1D array non log scaled
    :param H: Hurst exponent
    :param c: constant c
    :param show: boolean option to render the plot after the Hurst print out
    :return: axis object containing log log plot ax.show() will render the plot inline
    """

    _,ax = plt.subplots(figsize=(10,7))
    log_x = np.log10(x)
    log_y = np.log10(y)
    ax.plot(log_x,log_y,label='real') #plot empirical line
    
    lm = [c + n*H for n in log_x] # assume empirical solution for eq 4.8
    r2 = np.corrcoef(lm,y)[1][0]

    ax.plot(log_x,lm,label='fitted')
    ax.set_title('(R/S) Log Log Plot')
    ax.set_xlabel('Log R/S')
    ax.set_ylabel('Log Size')
    ax.text(0.2,0.8,"Y = {:.4f}X+{:.4f} \n $R^2$ = {:.3f}".format(H,c,r2),transform=ax.transAxes)
    ax.legend()

    if show: # option to render while running else return the axis object
        plt.show()

    return ax


if __name__ == '__main__':

    # Choose one way to load data

    # Use random_walk from existing hurst library.
    # Be aware that I have not verfied if this is a good pseudo-random number generator
    # No random number generator produces true random numbers. The series produced actually has long cycle, or memory, which causes the series to repeat.
    # Ideally, the pseudo-random generator reduces memory by following Peters FMH p.68 wherein the series are scrambled according to two other pseudo-random
    # series. The series should have mean zero and standard deviation of one.
    # series = np.array(random_walk(99999, cumprod=True))

    # OR

    # Load from SP500 dataset
    # series = np.genfromtxt('datasets/sp500.csv', delimiter=',')

    # OR

    # Load from SP500 dataset
    series = np.genfromtxt('datasets/dollar-yen-exchange-rate-historical-chart.csv', delimiter=',')[:,1] # this dataset is the best I can find to verify with Peters FMH. Expected values: H=0.642, c=-0.187

    # OR

    # load from quandl
    # series = quandl.get("WIKI/FB") # read data from quandl
    # series = series['Close'].to_numpy()

    # calculate log returns and AR(1) residuals as per Peters FMH p.62
    obv = __get_obv(series)
    series = __to_log_returns_series(series[:obv])
    series = __get_ar1_residuals(series)

    # Evaluate Hurst equation
    H, c, data = __compute_Hc(series)
    print("H={:.4f}, c={:.4f}".format(H,c)) # random walk should possess brownian motion Hurst statistics e.g. H=0.5

    #Log log plot
    __log_log_plot(data[0],data[1],H,c,show=True)
