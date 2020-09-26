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

def __compute_fbm(k_max, x=4/9, y=2/3, cdf=None, randomize=True):
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

    fbm = np.array(__construct_fbm_generator(0, 0, 1, 1, 1, k_max, x, y, cdf=cdf, randomize=randomize))
    x = fbm[:,0]
    y = fbm[:,1]
    x = np.delete(x, np.arange(0, x.size, 4))
    y = np.delete(y, np.arange(0, y.size, 4))

    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    return x, y 

def __construct_fbm_generator(x1, y1, x2, y2, k, k_max, x, y, cdf=None, randomize=True):
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

    p0, p1, p2, p3 = __construct_generator_from_initiator(x1, y1, x2, y2, x, y)

    if cdf != None:
        dT1 = cdf.diff_of_two_x(p0[0], p1[0])
        dT2 = cdf.diff_of_two_x(p1[0], p2[0])
        dT3 = cdf.diff_of_two_x(p2[0], p3[0])
        p1[0] = p0[0] + (x2 - x1) * (dT1 / (dT1 + dT2 + dT3))
        p2[0] = p1[0] + (x2 - x1) * (dT2 / (dT1 + dT2 + dT3))
        p3[0] = p2[0] + (x2 - x1) * (dT3 / (dT1 + dT2 + dT3))

    if randomize:
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

    if (k == k_max):
        return [p0, p1, p2, p3]

    fbm = __construct_fbm_generator(p0[0], p0[1], p1[0], p1[1], k+1, k_max, x, y, cdf, randomize)
    fbm = np.append(fbm, __construct_fbm_generator(p1[0], p1[1], p2[0], p2[1], k+1, k_max, x, y, cdf, randomize), axis=0)
    fbm = np.append(fbm, __construct_fbm_generator(p2[0], p2[1], p3[0], p3[1], k+1, k_max, x, y, cdf, randomize), axis=0)

    return fbm

def __simulate_bmmt(k_max, M=[0.6, 0.4], x=4/9, y=2/3, randomize=True):
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

    return __compute_fbm(k_max, x, y, cdf=cdf, randomize=True)

def __construct_generator_from_initiator(x1, y1, x2, y2, x, y):
    delta_x = x2 - x1
    delta_y = y2 - y1

    p0 = [x1, y1]
    p1 = [x1 + delta_x * x, y1 + delta_y * y]
    p2 = [p1[0] + delta_x * (1 - 2 * x), p1[1] - delta_y * ((2 * y) - 1)]
    p3 = [x2, y2]

    return p0, p1, p2, p3
