import numpy as np
import math
from fma.mmar.multiplicative_cascade import MutiplicativeCascade
from fma.mmar.trading_time_cdf import TradingTimeCDF
    
def __compute_multiplicative_cascade(k_max, M, randomize=False):
    """
    Helper function for __cascade
    :param k_max: max depth of the recursion tree
    :param M: array m0, m1, ..., mb where sum(M) = 1
    :param randomize: whether or not to shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
    :return: [x, ...], [y, ...] corresponding to multiplicative cascade y coordinates
    """
    c = MutiplicativeCascade(k_max, M, randomize)
    
    return c.cascade()

def __compute_fbm(k_max, x=4/9, y=2/3, cdf=None):
    """
    y^(1/H) = x
    :param k_max: max depth of the recursion tree
    :param x: x coord of point P in generator. Brownian motion x=4/9 if y=2/3
    :param y: y coord of point P in generator. Brownian motion x=4/9 if y=2/3
    :return: x, y of fbm timeseries
    """
    lhs = math.log(y, y)
    rhs = math.log(x,y)

    H = 1/rhs
    print("Creating fbm with H={:.4f}".format(H)) 

    fbm = np.array(__construct_fbm_generator(0, 0, 1, 1, 1, k_max, x, y, cdf))
    x = fbm[:,0]
    y = fbm[:,1]
    x = np.delete(x, np.arange(0, x.size, 4))
    y = np.delete(y, np.arange(0, y.size, 4))

    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    return x, y 

def __construct_fbm_generator(x1, y1, x2, y2, k, k_max, x, y, cdf):
    """
    H = 1/2
         ________
    2/3 |__     /|  
        |  /\  / | 1
        | /| \/__| 1/3 
        |/_|__|__|      
        4/9  5/9 
            1

    y^(1/H) + (2y - 1)^(1/H) + y^(1/H) = 1
    y = x^H
    
    :param x1: left x coord of initiator
    :param y1: left y coord of initiator
    :param x2: right x coord of initiator
    :param y2: right y coord of initiator
    :param k: current branch of the recursion tree
    :param k_max: max depth of the recursion tree
    :param x: x coord of point P in generator. Brownian motion x=4/9 if y=2/3
    :param y: y coord of point P in generator. Brownian motion x=4/9 if y=2/3
    :param cdf: cdf of trading time
    :return: fbm generator
    """

    delta_x = x2 - x1
    delta_y = y2 - y1

    p0 = [x1, y1]
    p1 = [x1 + delta_x * x, y1 + delta_y * y]
    p2 = [p1[0] + delta_x * (1 - 2 * x), p1[1] - delta_y * ((2 * y) - 1)]
    p3 = [x2, y2]

    i1_lower = cdf.find_interval(p0[0]) - 1
    i1_upper = cdf.find_interval(p1[0]) - 1

    i2_lower = cdf.find_interval(p1[0]) - 1
    i2_upper = cdf.find_interval(p2[0]) - 1

    dT1 = cdf.diff_at_index(i1_lower, i1_upper)
    dT2 = cdf.diff_at_index(i2_lower, i2_upper)

    print("dt1 {:.5f}, dt2 {:.5f}".format(dT1, dT2))
    p1_t = p1[0]
    p2_t = p2[0]

    p1[0] = p0[0] + dT1
    p2[0] = p1[0] + dT2

    w0 = p1[0] - p0[0]
    h0 = p1[1] - p0[1]

    w1 = p2[0] - p1[0]
    h1 = p2[1] - p1[1]

    w2 = p3[0] - p2[0]
    h2 = p3[1] - p2[1]

    segments = [[w0, h0], [w1, h1], [w2, h2]]
    np.random.shuffle(segments)

    x_0 = p0[0]
    y_0 = p0[1]

    x_1 = x_0 + segments[0][0]
    y_1 = y_0 + segments[0][1]

    x_2 = x_1 + segments[1][0]
    y_2 = y_1 + segments[1][1]

    x_3 = x_2 + segments[2][0]
    y_3 = y_2 + segments[2][1]

    p0 = [x_0, y_0]
    p1 = [x_1, y_1]
    p2 = [x_2, y_2]
    p3 = [x_3, y_3]
    print("{:.6f} - {:.6f} -> {:.6f} - {:.6f}; i1_l: {:.6f}".format(p1_t, p2_t, p1[0], p2[0], i1_lower))

    if (k == k_max):
        return [p0, p1, p2, p3]
    
    fbm = __construct_fbm_generator(p0[0], p0[1], p1[0], p1[1], k+1, k_max, x, y, cdf)
    fbm = np.append(fbm, __construct_fbm_generator(p1[0], p1[1], p2[0], p2[1], k+1, k_max, x, y, cdf), axis=0)
    fbm = np.append(fbm, __construct_fbm_generator(p2[0], p2[1], p3[0], p3[1], k+1, k_max, x, y, cdf), axis=0)

    return fbm

def __simulate_bmmt(k_max, M=[0.6, 0.4], x=4/9, y=2/3, randomize=False):
    """
    y^(1/H) = x
    :param k_max: max depth of the recursion tree
    :param M: array m0, m1, ..., mb where sum(M) = 1
    :param x: x coord of point P in generator. Brownian motion x=4/9 if y=2/3
    :param y: y coord of point P in generator. Brownian motion x=4/9 if y=2/3
    :param randomize: whether or not to shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
    :return: x, y of fbm timeseries
    """
    lhs = math.log(y, y)
    rhs = math.log(x,y)

    H = 1/rhs
    print("Creating fbm with H={:.4f}".format(H)) 

    cdf = TradingTimeCDF(k_max, M, randomize)
    cdf.compute_cdf()

    print("y coords cdf:")
    print(cdf.y)

    return __compute_fbm(k_max, x, y, cdf)

def __combine_fbm_and_trading_time(x_brownian, x_trading, y_trading):
    """
    :param fbm: fractional brownian motion "Mother"
    :param x_trading: x coords of trading time "Father"
    :param y_trading: y coords of trading time "Father"
    :return: brownian motion in multifractal trading time
    """
    for i in range(1, len(x_brownian) - 1, 3):
        xb_l = x_brownian[i]
        xb_u = x_brownian[i+1]
        xb_mid = (xb_u + xb_l) / 2
        xb_delta = xb_u - xb_l

        xt_l = x_trading[math.floor(i / 2)]
        xt_u = x_trading[math.floor(i / 2) + 1]

        yt_l = y_trading[math.floor(i / 2)]
        yt_u = y_trading[math.floor(i / 2) + 1]

        slope_abs = abs((yt_u - yt_l) / (xt_u - xt_l))
        
        xt_delta = slope_abs * xb_delta # Use this ?
        yt_delta = yt_u - yt_l # Use this ?

        xb_l_new = xb_mid - (xt_delta / 2)
        xb_u_new = xb_mid + (xt_delta / 2)

        x_brownian[i] = xb_l_new
        x_brownian[i + 1] = xb_u_new
    
    return x_brownian

