import numpy as np
from fma.mmar.multiplicative_cascade import MutiplicativeCascade
from scipy import interpolate

class TradingTimeCDF:
    def __init__(self, k_max, M, randomize=False):
        self.k_max = k_max
        self.M = M
        self.randomize = randomize
        self.cdf = None
        self.x = []
        self.y = []
    
    def create_trading_time_cdf(self):
        x, y = self._create_trading_time_cdf(self.k_max, self.M, self.randomize)
        self.x = np.append(x, 1)
        self.y = np.append(y, 1)

        self.cdf = interpolate.interp1d(self.x, self.y)

    def _create_trading_time_cdf(self, k_max, M, randomize=False):
        """
        :param k_max: max depth of the recursion tree
        :param M: array m0, m1, ..., mb where sum(M) = 1
        :param randomize: whether or not to shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
        :return: [x, ...], [y, ...] corresponding to cdf of trading time
        """
        c1 = MutiplicativeCascade(k_max, M, randomize)
        c1.cascade()
        x1, y1 = c1.x, c1.y

        c2 = MutiplicativeCascade(k_max, M, randomize)
        c2.cascade()
        x2, y2 = c2.x, c2.y

        return np.cumsum(y1 * (1 / len(y1))), np.cumsum(y2 * (1 / len(y2)))