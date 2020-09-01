from fractalmarkets.mmar.multiplicative_cascade import MutiplicativeCascade

def test_multiplicative_cascade():
    c = MutiplicativeCascade(1, [0.6, 0.4], False)
    c.cascade()

    assert all(a == b for a, b in zip(c.data[:,0], [0, 0.5, 1]))
    assert all(a == b for a, b in zip(c.data[:,1], [0, 1.2, 0.8]))
