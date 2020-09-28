from fma.mmar.multiplicative_cascade import MutiplicativeCascade

def test_multiplicative_cascade():
    c = MutiplicativeCascade(1, [0.6, 0.4], False)
    c.cascade()

    x, y = c.x, c.y

    assert all(a == b for a, b in zip(x, [0, 0.5, 1]))
    assert all(a == b for a, b in zip(y, [0, 1.2, 0.8]))
