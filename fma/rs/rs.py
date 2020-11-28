from fma.rs.metrics import get_obv, to_log_returns_series, get_ar1_residuals, compute_Hc
from fma.rs.plots import log_log_plot

class RS:
    def __init__(self, y):
        # require type(y) == 'numpy.ndarray'
        if type(y).__module__ == 'numpy':
            self.y = y
        else:
            raise TypeError("{} is not a supported type for y. Supported type for y is 'numpy.ndarray'. Convert a list to ndarray with np.array(list).".format(type(y)))

    def get_H(self):
        obv = get_obv(self.y)
        logs = to_log_returns_series(self.y[:obv])
        residuals = get_ar1_residuals(logs)
        
        return compute_Hc(residuals)

    def plot_vstat(self):
        H, c, data = self.get_H()

        log_log_plot(data[0], data[1], H, c, V_stat=True)
        