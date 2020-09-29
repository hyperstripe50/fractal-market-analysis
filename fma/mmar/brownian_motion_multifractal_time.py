from fma.mmar.brownian_motion import BrownianMotion
import numpy as np
from fma.mmar.trading_time_cdf import TradingTimeCDF

class BrownianMotionMultifractalTime(BrownianMotion):
    def __init__(self, k_max, x, y, randomize=False, M=[0.6, 0.4]):
        """
         y^(1/H) = x
        :param k_max: max depth of the recursion tree
        :param x: x coord of point P in generator. Brownian motion x=4/9 if y=2/3
        :param y: y coord of point P in generator. Brownian motion x=4/9 if y=2/3
        :param randomize: shuffle symmetric generator and shuffle M before assigning mass to child cells. See page 13 of "A Multifractal Model of Asset Returns" 1997
        :param M: array m0, m1, ..., mb where sum(M) = 1
        """
        super().__init__(k_max, x, y, randomize)
        self.M = M

    def simulate(self):
        """
        :return: x, y of bownian motion in multifractal time timeseries
        """

        trading_time = TradingTimeCDF(self.k_max, self.M, self.randomize)
        trading_time.create_trading_time_cdf()

        fbm = np.array(super()._simulate_bm_recursively(0, 0, 1, 1, 1, self.k_max, self.x, self.y, cdf=trading_time.cdf, randomize=self.randomize))

        x = fbm[:,0]
        y = fbm[:,1]
        x = np.delete(x, np.arange(0, x.size, 4))
        y = np.delete(y, np.arange(0, y.size, 4))
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)

        return x, y