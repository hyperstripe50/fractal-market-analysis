import numpy as np
from fractalmarkets.mmar.multiplicative_cascade import MutiplicativeCascade
from scipy import interpolate

class TradingTimeCDF:
    def __init__(self, k_max, M, randomize=False):
        self.k_max = k_max
        self.M = M
        self.randomize = randomize
        self.cdf = None
        self.cascade = None
        self.data = []
    
    def create_trading_time_cdf(self):
        x, y = self._create_trading_time_cdf(self.k_max, self.M, self.randomize)
        self.data = np.stack([x, y], axis=1)
        
        self.cdf = interpolate.interp1d(x, y)

    def _create_trading_time_cdf(self, k_max, M, randomize=False):
        """
        :param k_max: max depth of the recursion tree
        :param M: array m0, m1, ..., mb where sum(M) = 1
        :param randomize: whether or not to shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
        :return: [x, ...], [y, ...] corresponding to cdf of trading time
        """
        self.cascade = MutiplicativeCascade(k_max, M, randomize)
        self.cascade.cascade()

        return self.cascade.data[:,0], np.cumsum(self.cascade.data[:,1]) / (len(self.cascade.data) - 1)