import matplotlib.pyplot as plt; plt.style.use('ggplot')
from fma.mmar.timeseries import __compute_multiplicative_cascade

if __name__ == '__main__':
    x, y = __compute_multiplicative_cascade(4, [0.6, 0.4], False)
    plt.step(x, y, where='post')
    plt.ylim(bottom=0)
    plt.xlim(0)
    plt.xticks(x)
    plt.show()