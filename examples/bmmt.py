import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np 
import math
import argparse
from scipy import interpolate
from fma.mmar.brownian_motion import BrownianMotion

from fma.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='get options for bmmt generator')
    parser.add_argument('--iters',dest='iters',type=int,help='number of iterations',default=9)
    parser.add_argument('--alloc',dest='alloc',type=float, nargs='+', help='M-1 allocations of mass',default=[0.6])
    parser.add_argument('--randomize_segments',dest='randomize_segments',type=bool,nargs='?',help='Randomize segments',const=True)
    parser.add_argument('--randomize_time',dest='randomize_time',type=bool,nargs='?',help='Randomize time',const=True)
    parser.add_argument('--px',dest='px',type=float,help='x coord of first break point in generator. (0, 1/2)',default=4/9)
    parser.add_argument('--py',dest='py',type=float,help='y coord of first break point in generator. (0, 1)',default=2/3)
    parser.add_argument('--non-log-returns',dest='nonlog',type=bool,nargs='?',help='Display log returns',const=True)

    args=parser.parse_args()

    M = np.append(args.alloc, 1 - np.sum(args.alloc))

    bmmt = BrownianMotionMultifractalTime(args.iters, x=args.px, y=args.py, randomize_segments=args.randomize_segments, randomize_time=args.randomize_time, M=M)
    data = bmmt.simulate()

    y1 = data[:,1]
    if (args.nonlog):
        y1 = [ math.pow(10, y) for y in data[:,1] ]

    f = interpolate.interp1d(data[:,0], y1)

    y1 = [f(x) for x in np.arange(0, 1, .00001)]
    x1 = np.linspace(0, 1, len(y1), endpoint=True)

    y2 = [b - a for a, b in zip(y1[:-1], y1[1:])]

    fig, axs = plt.subplots(2)
    fig.suptitle('MMAR')

    axs[0].plot(x1, y1, 'b-')
    axs[1].plot(x1[:-1], y2)

    z1 = np.array(y1)
    z2 = np.array([0] * len(y1))

    axs[0].fill_between(x1, y1, 0,
                 where=(z1 >= z2),
                 alpha=0.30, color='green', interpolate=True)
    
    axs[0].fill_between(x1, y1, 0,
                 where=(z1 < z2),
                 alpha=0.30, color='red', interpolate=True)

    plt.show()