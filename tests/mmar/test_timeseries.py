from fma.mmar.timeseries import __compute_multiplicative_cascade

def test_multiplicative_cascade():
    x, y = __compute_multiplicative_cascade(1, [0.6, 0.4], False)

    assert all(a == b for a, b in zip(x, [0, 0.5, 1]))
    assert all(a == b for a, b in zip(y, [1.2, 0.8, 0.8]))
