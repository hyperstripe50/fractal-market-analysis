import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np 
import math
import argparse
from fma.mmar.brownian_motion_multifractal_time import BrownianMotionMultifractalTime

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='get options for bmmt generator')
    parser.add_argument('--iters',dest='iters',type=int,help='number of iterations',default=9)
    parser.add_argument('--alloc',dest='alloc',type=float, nargs='+', help='M-1 allocations of mass',default=[0.6])
    parser.add_argument('--randomize',dest='randomize',type=bool,nargs='?',help='Randomize order of M',const=True)
    parser.add_argument('--x',dest='px',type=float,help='x coord of first break point in generator. (0, 1/2)',default=4/9)
    parser.add_argument('--y',dest='py',type=float,help='y coord of first break point in generator. (0, 1)',default=2/3)
    parser.add_argument('--non-log-returns',dest='nonlog',type=bool,nargs='?',help='Display log returns',const=True)

    args=parser.parse_args()

    M = np.append(args.alloc, 1 - np.sum(args.alloc))

    bmmt = BrownianMotionMultifractalTime(args.iters, x=args.px, y=args.py, randomize=args.randomize, M=M)
    x1, y1 = bmmt.simulate()

    if (args.nonlog):
        y1 = [ math.pow(10, y) for y in y1 ]

    plt.plot(x1,y1, 'b-')

    z1 = np.array(y1)
    z2 = np.array([0] * len(y1))

    plt.fill_between(x1, y1, 0,
                 where=(z1 >= z2),
                 alpha=0.30, color='green', interpolate=True)

    plt.show()