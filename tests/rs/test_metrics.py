from fractalmarkets.rs.metrics import compute_ers
from math import log10

def test_ers():
    expected = 1.4880
    actual   = round(log10(compute_ers(650)), 3)

    # for lower values of n, i.e. compute_ers(x) where x < 650, the actual diverges slightly from expected
    assert actual == expected