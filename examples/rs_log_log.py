from fma.rs.rs import RS
import numpy as np
from fma.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime
from fma.mmar.brownian_motion import BrownianMotion
from scipy import interpolate

if __name__ == '__main__':
    # Load from Dollar Yen historical exchange rate
    # series = np.genfromtxt('datasets/dollar-yen-exchange-rate-historical-chart.csv', delimiter=',')[::1,1] # this dataset is the best I can find to verify with Peters FMH. Expected values: H=0.642, c=-0.187
    bmmt = BrownianMotionMultifractalTime(9, x=4/9, y=0.9, randomize_segments=False, randomize_time=False, M=[0.6, 0.4])
    data = bmmt.simulate()
    print("Expected H {}".format(bmmt.get_H()))

    series = np.array([ z + 10 for z in data[:,1] ])
    rs = RS(series)
    (H, c, data) = rs.get_H()

    print("H={:.4f}, c={:.4f}".format(H,c)) # random walk should possess brownian motion Hurst statistics e.g. H=0.5

    rs.plot_vstat()