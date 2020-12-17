import numpy as np
import math

def to_log_returns_series(x):
    """
    :param x: 1D array of numbers
    :return: 1D array of log returns
    """
    log_returns = np.log(x[1:]/x[:-1])

    return log_returns

def get_obv(x):
    """
    :param x: 1D array
    :return: number of observations to the lower 100 + 2. Since AR(1) residual is passed on to R/S, we lose two points along the way.
             Thus, we need to start with two more observations than we wish to pass to the R/S section.
    """
    obv = math.floor((len(x)-1)/100)*100 + 2

    return obv

def get_ar1_residuals(x):
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

def compute_ers(n):
    """
    :param n: log(number of observations)
    :return: E(R/S) as per Peters FMH p. 71 derived correction. Test values on p. 115
    E(R/S_n) = ((n - 0.5) / n) * ( n * pi/2 )^-0.5 * sum(r=1 -> n-1, sqrt((n-r)/r))
    """

    return ((n - 0.5) / n) * math.pow(n * (math.pi / 2), -0.5) * np.sum(np.array([ math.sqrt((n-r)/r) for r in range(1, n-1) ]))

def get_rs(x):
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

def get_rs_data(x):
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
                rs.append(get_rs(x[start:start + n]))
            RS.append(np.mean(rs))
            N.append(n)

    return [np.array(N).astype(int), np.array(RS)]

def get_Hc(rs_data, max_n=-1):
    N = rs_data[0] if max_n == -1 else rs_data[0][np.where(rs_data[0] <= max_n)]
    RS = rs_data[1][:len(N)]
    A = np.vstack([np.log10(N), np.ones(len(N))]).T # y = Ap, where A = [[x 1]] and p = [[m], [c]]
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0] # slope (Hurst exponent), intercept (constant); WRT Peters FMH p. 56 eq 4.7 (R/S)_n = c*n^H

    return H, c