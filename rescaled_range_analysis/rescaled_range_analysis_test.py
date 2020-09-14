from rescaled_range_analysis import __compute_ers
from math import log10

def test_ers():
    expected = 1.4880
    actual   = round(log10(__compute_ers(650)), 3)

    # for lower values of n, i.e. __compute_ers(x) where x < 650, the actual diverges slightly from expected
    assert actual == expected