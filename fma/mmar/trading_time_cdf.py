import numpy as np
from fma.mmar.multiplicative_cascade import MutiplicativeCascade
import math

class TradingTimeCDF:
    def __init__(self, k_max, M, randomize=False):
        self.k_max = k_max
        self.M = M
        self.randomize = randomize
        self.x = []
        self.y = []
    
    def compute_cdf(self):
        x, y = self.__compute_trading_time(self.k_max, self.M, self.randomize)
        self.x = np.append(x, 1)
        self.y = np.append(y, 1)

    def find_interval(self, value):
        idx = np.searchsorted(self.x, value, side="left")

        if idx > 0 and (math.fabs(value - self.x[idx-1]) < math.fabs(value - self.x[idx])):
            return idx
        elif idx == 0:
            return 1
        else:
            return idx

    def diff_at_index(self, lower, upper):
        return self.y[upper] - self.y[lower]

    def sample_cdf_at_index(self, index):
        return self.y[index]

    def get_slope(self, lower, upper):
        return (self.y[upper] - self.y[lower]) / (self.x[upper] - self.x[lower])

    def __compute_trading_time(self, k_max, M, randomize=False):
        """
        :param k_max: max depth of the recursion tree
        :param M: array m0, m1, ..., mb where sum(M) = 1
        :param randomize: whether or not to shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
        :return: [x, ...], [y, ...] corresponding to cdf of trading time
        """
        c = MutiplicativeCascade(k_max, M, randomize)

        x, y = c.cascade()
        x2, y2 = c.cascade()
        
        return np.cumsum(y * (1 / len(y))), np.cumsum(y2 * (1 / len(y2)))