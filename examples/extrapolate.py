import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np 
import math
import argparse
import pandas as pd
from scipy import interpolate
from fma.mmar.brownian_motion import BrownianMotion
from scipy import optimize
from fma.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime

def inverse_corr(z, *params):
    px, py, m0 = z
    m1 = 1 - m0 

    target_y = params

    bmmt = BrownianMotionMultifractalTime(5, x=px, y=py, randomize_segments=True, randomize_time=True, M=[m0, m1])
    data = bmmt.simulate()
    f = interpolate.interp1d(data[:,0], data[:,1])
    y = [f(z) for z in np.arange(0, 1, .001)]
    y = np.array(y)
    y_diff = (np.diff(y) / y[1:] * 100)
    target_y_diff = (np.diff(target_y) / target_y[1:] * 100)
    return np.sqrt(np.mean(((y_diff - target_y_diff)**2)))

def callback(x, f, context):
    print(x, flush=True)
    print(f, flush=True)
    print(context, flush=True)
    print()

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='get options for bmmt generator')
    parser.add_argument('--alloc',dest='alloc',type=float, nargs='+', help='M-1 allocations of mass',default=[0.6])
    parser.add_argument('--x',dest='px',type=float,help='x coord of first break point in generator. (0, 1/2)',default=4/9)
    parser.add_argument('--y',dest='py',type=float,help='y coord of first break point in generator. (0, 1)',default=2/3)

    args=parser.parse_args()

    M = np.append(args.alloc, 1 - np.sum(args.alloc))

    bmmt = BrownianMotionMultifractalTime(5, x=0.41, y=0.64, randomize_segments=True, randomize_time=True, M=[0.68, 0.32])
    data = bmmt.simulate()

    f = interpolate.interp1d(data[:,0], data[:,1])

    y = [f(z) for z in np.arange(0, 1, .001)]
    y = np.array(y)

    # df = pd.read_csv("/Users/jona/Downloads/VOO.csv")
    # y = df['Close'].to_list()[:1000]
    res = optimize.dual_annealing(inverse_corr, callback=callback, x0=(args.px, args.py, args.alloc[0]), bounds=[(0.2, 0.49), (0.55, 0.8), (0.1, 0.9)], args=y)
    print("-------- results -----------")
    print("params: " + str(res.x))
    print("corr: " + str(res.fun))
