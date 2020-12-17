from fractalmarkets.rs.metrics import get_obv, to_log_returns_series, get_ar1_residuals, get_rs_data, get_Hc
from fractalmarkets.rs.plots import log_log_plot
import numpy as np 

class RS:
    def __init__(self, y):
        # require type(y) == 'numpy.ndarray'
        if type(y).__module__ == 'numpy':
            self.y = y
        else:
            raise TypeError("{} is not a supported type for y. Supported type for y is 'numpy.ndarray'. Convert a list to ndarray with np.array(list).".format(type(y)))

    def get_Hc(self, max_n = -1):
        return get_Hc(self.analyze(), max_n=max_n)
    
    def analyze(self):
        obv = get_obv(self.y)
        logs = to_log_returns_series(self.y[:obv])
        residuals = get_ar1_residuals(logs)

        return get_rs_data(residuals)

    def get_cycles(self, split_validations=2):
        tmpy = self.y
        yarr = np.split(self.y, split_validations)

        cycles = np.array([])
        for y in yarr:
            tmp_cycles = np.array([])
            self.y = y
            data = self.analyze()

            vstaty = data[1]/np.sqrt(data[0])
            vstatd = np.stack([data[0], vstaty], axis=1)
            for a, b in zip(vstatd[:-1], vstatd[1:]):
                if b[1] > a[1]:
                    tmp_cycles = np.append(tmp_cycles, [round(b[0], 0)])
            
            cycles = np.union1d(cycles, tmp_cycles)
        
        self.y = tmpy

        return cycles

    def plot_vstat(self, max_n = -1):
        data = self.analyze()
        H, c = self.get_Hc(max_n=max_n)

        N = data[0] if max_n == -1 else data[0][np.where(data[0] <= max_n)]
        RS = data[1][:len(N)]

        log_log_plot(N, RS, H, c, V_stat=True)
        