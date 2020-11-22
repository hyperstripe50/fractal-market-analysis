import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from fma.mmar.brownian_motion import BrownianMotion
import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='get options for bmmt generator')
    parser.add_argument('--iters',dest='iters',type=int,help='number of iterations',default=9)
    parser.add_argument('--randomize_segments',dest='randomize_segments',type=bool,nargs='?',help='Randomize segments',const=True)
    parser.add_argument('--px',dest='px',type=float,help='x coord of first break point in generator. (0, 1/2)',default=4/9)
    parser.add_argument('--py',dest='py',type=float,help='y coord of first break point in generator. (0, 1)',default=2/3)

    args=parser.parse_args()

    bm1 = BrownianMotion(args.iters, args.px, args.py, randomize_segments=args.randomize_segments)
    data = bm1.simulate()

    plt.plot(data[:,0], data[:,1], 'b-')

    plt.show()