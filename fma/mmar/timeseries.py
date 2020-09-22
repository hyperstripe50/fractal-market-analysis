import numpy as np
import math

def __compute_multiplicative_cascade(k_max, M, randomize=False):
    """
    Helper function for __cascade
    :param k_max: max depth of the recursion tree
    :param M: array m0, m1, ..., mb where sum(M) = 1
    :param randomize: whether or not to shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
    :return: [x, ...], [y, ...] corresponding to multiplicative cascade y coordinates
    """
    y = __cascade(1, 1, 1, k_max, M, randomize) 
    x = np.linspace(0, 1, num=len(y), endpoint=False)
    
    y = np.insert(y, 0, 0)  # y(0) = 0
    x = np.append(x, 1)     # duplicate last to draw proper graph

    return x, y

def __cascade(x, y, k, k_max, M, randomize=False):
    """
    :param x: width of current cell
    :param y: height of current cell
    :param k: current branch of the recursion tree
    :param k_max: max depth of the recursion tree
    :param M: array m0, m1, ..., mb where sum(M) = 1
    :param randomize: whether or not to shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
    :return: [y, ...] corresponding to multiplicative cascade y coordinates
    """
    a = x * y
    x_next = x / len(M)

    M_shuffle = np.copy(M)
    if randomize:
        np.random.shuffle(M_shuffle)
    else:
        M_shuffle = M_shuffle
        
    y_i = np.array([])
    if (k == k_max):
        for m in M_shuffle:
            y_i = np.append(y_i, (m * a) / x_next)

        return y_i

    for m in M_shuffle:
        y_i = np.append(y_i, __cascade(x_next, (m * a) / x_next, k + 1, k_max, M, randomize))
    
    return y_i

def __compute_trading_time(k_max, M, randomize=False):
    """
    :param k_max: max depth of the recursion tree
    :param M: array m0, m1, ..., mb where sum(M) = 1
    :param randomize: whether or not to shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
    :return: [x, ...], [y, ...] corresponding to cdf of trading time
    """
    x, y = __compute_multiplicative_cascade(k_max, M, randomize)
    x2, y2 = __compute_multiplicative_cascade(k_max, M, randomize)

    return np.cumsum(y * (1 / len(y))), np.cumsum(y2 * (1 / len(y2)))

def __compute_fbm(k_max, x=4/9, y=2/3):
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

    fbm = __construct_fbm_generator(0, 0, 1, 1, 1, k_max, x, y)
    x = fbm[:,0]
    y = fbm[:,1]
    x = np.delete(x, np.arange(0, x.size, 4))
    y = np.delete(y, np.arange(0, y.size, 4))

    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    return x, y 

def __construct_fbm_generator(x1, y1, x2, y2, k, k_max, x, y):
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
    
    :param x1: left x coord
    :param y1: left y coord
    :param x2: right x coord
    :param y2: right y coord
    :param k: current branch of the recursion tree
    :param k_max: max depth of the recursion tree
    :param x: x coord of point P in generator. Brownian motion x=4/9 if y=2/3
    :param y: y coord of point P in generator. Brownian motion x=4/9 if y=2/3
    :return: fbm generator
    """

    delta_x = x2 - x1
    delta_y = y2 - y1

    p0 = [x1, y1]
    p1 = [x1 + delta_x * x, y1 + delta_y * y]
    p2 = [p1[0] + delta_x * (1 - 2 * x), p1[1] - delta_y * ((2 * y) - 1)]
    p3 = [x2, y2]

    if (k == k_max):
        return [p0, p1, p2, p3]
    
    fbm = __construct_fbm_generator(p0[0], p0[1], p1[0], p1[1], k+1, k_max, x, y)
    fbm = np.append(fbm, __construct_fbm_generator(p1[0], p1[1], p2[0], p2[1], k+1, k_max, x, y), axis=0)
    fbm = np.append(fbm, __construct_fbm_generator(p2[0], p2[1], p3[0], p3[1], k+1, k_max, x, y), axis=0)

    return fbm

