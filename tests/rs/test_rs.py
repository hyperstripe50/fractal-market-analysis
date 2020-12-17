from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from fractalmarkets.rs.rs import RS

def test_mean_reverting():
    s = log(cumsum(randn(100000))+1000)

    rs = RS(s)
    H, c = rs.get_Hc()

    assert 0.45 < round(H, 2) < 0.55