import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np
import math
from fractalmarkets.rs.metrics import compute_ers

def log_log_plot(x,y,H,c,show=True,V_stat=True):

    """
    :param x: 1D array non log scaled
    :param y: 1D array non log scaled
    :param H: Hurst exponent
    :param c: constant c
    :param show: bool option to render the plot after the Hurst print out
    :param V_stat: bool option to add a subplot with the V statistic plotted against the log size
    :return: axis object containing log log plot ax.show() will render the plot inline
    """

    plt.figure(figsize=(10,20))
    plt.subplots_adjust(hspace=0.5)

    ax = plt.subplot(2,1,1)

    log_x = np.log10(x)
    log_y = np.log10(y)
    ax.plot(log_x,log_y,'ro-',label='R/S') # plot empirical line

    # annotate breaks
    for a,b in zip(log_x, log_y):
        label = "{:.0f}".format(math.pow(10, a))

        plt.annotate(label,
                     (a,b),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')

    ey = [ compute_ers(n) for n in x]
    log_ey = [ math.log10(n) for n in ey ]

    lm=[c + n*H for n in log_x] # assume empirical solution for eq 4.8
    r2=np.corrcoef(lm,log_y)[1][0]

    ax.plot(log_x,log_ey,'g--',label='E(R/S)')
    ax.plot(log_x,lm,'b--',label='Fitted Empirical')
    ax.set_title('(R/S) Log Log Plot')
    ax.set_xlabel('Log Size')
    ax.set_ylabel('Log R/S')
    ax.text(0.2,0.8,"(fitted) Y = {:.4f}X{}{:.4f} \n $R^2$ = {:.3f}".format(H,"+" if c>0 else "",c,r2),transform=ax.transAxes)
    ax.legend()

    if V_stat:
        ax_v=plt.subplot(2,1,2)
        ax_v.plot(log_x,y/np.sqrt(x),'k-',label='V Stat')
        ax_v.plot(log_x,ey/np.sqrt(x), 'g--', label='E(R/S)')
        ax_v.set_title('V Statistic Plot')
        ax_v.set_xlabel('Log Size')
        ax_v.set_ylabel('V Stat')

        # annotate breaks
        for a,b in zip(log_x, y/np.sqrt(x)):
            label = "{:.0f}".format(math.pow(10, a))

            plt.annotate(label,
                         (a,b),
                         textcoords="offset points",
                         xytext=(0,10),
                         ha='center')

        ax_v.legend()

    if show: # option to render while running else return the axis object
        plt.show()

    axes = [ax]
    if ax_v is not None: #need a way to keep track of all the axes objects for our plots
        axes.append(axes)

    return ax