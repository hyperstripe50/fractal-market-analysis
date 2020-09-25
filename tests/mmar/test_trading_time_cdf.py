from fma.mmar.trading_time_cdf import TradingTimeCDF

def test_find_interval_0_is_1():
    cdf = TradingTimeCDF(1, [0.6, 0.4], randomize=False)
    cdf.compute_cdf() # x = [0., 0.4, 0.66666667, 1.]

    assert cdf.find_interval(0) == 1

def test_find_interval_first():
    cdf = TradingTimeCDF(1, [0.6, 0.4], randomize=False)
    cdf.compute_cdf()
    # x = [0., 0.4, 0.66666667, 1.]
    # I_1 = (0, 0.4]
    # I_2 = (0.4 - 0.6667]
    # I_3 = (0.667 - 1]

    assert cdf.find_interval(0.3) == 1

def test_find_interval_middle():
    cdf = TradingTimeCDF(1, [0.6, 0.4], randomize=False)
    cdf.compute_cdf()
    # x = [0., 0.4, 0.66666667, 1.]
    # I_1 = (0, 0.4]
    # I_2 = (0.4 - 0.6667]
    # I_3 = (0.667 - 1]

    assert cdf.find_interval(0.5) == 2

def test_find_interval_last():
    cdf = TradingTimeCDF(1, [0.6, 0.4], randomize=False)
    cdf.compute_cdf()
    # x = [0., 0.4, 0.66666667, 1.]
    # I_1 = (0, 0.4]
    # I_2 = (0.4 - 0.6667]
    # I_3 = (0.667 - 1]

    assert cdf.find_interval(.7) == 3

def test_find_interval_end():
    cdf = TradingTimeCDF(1, [0.6, 0.4], randomize=False)
    cdf.compute_cdf()
    # x = [0., 0.4, 0.66666667, 1.]
    # I_1 = (0, 0.4]
    # I_2 = (0.4 - 0.6667]
    # I_3 = (0.667 - 1]

    assert cdf.find_interval(1) == 3