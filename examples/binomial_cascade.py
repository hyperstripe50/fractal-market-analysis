import argparse
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from fma.mmar.timeseries import __compute_multiplicative_cascade

if __name__ == '__main__':

    parser=argparse.ArgumentParser(description='get options for multifractal measure generator')
    parser.add_argument('-iters',dest='iters',type=int,help='number of iterations',default=4)
    parser.add_argument('-hurst',dest='H',type=float,help='Hurst exponent',default=0.6)
    parser.add_argument('--randomize',dest='shuffle',type=bool,help='Shuffle parameter',default=False)
    args=parser.parse_args()

    x, y = __compute_multiplicative_cascade(args.iters, [args.H, 1-args.H], args.shuffle)
    plt.step(x, y, where='post')
    plt.ylim(bottom=0)
    plt.xlim(0)
    # plt.xticks(x)
    plt.show()