from rescaled_range_analysis import __compute_ers, __compute_multiplicative_cascade
from math import log10

def test_ers():
    expected = 1.4880
    actual   = round(log10(__compute_ers(650)), 3)

    # for lower values of n, i.e. __compute_ers(x) where x < 650, the actual diverges slightly from expected
    assert actual == expected

def test_multiplicative_cascade():
    x, y = __compute_multiplicative_cascade(1, [0.6, 0.4], False)

    assert all(a == b for a, b in zip(x, [0, 1]))
    assert all(a == b for a, b in zip(y, [1.2, 0.8]))
