from fma.mmar.trading_time_cdf import TradingTimeCDF

def test_diff_at_index_with_pct():
    cdf = TradingTimeCDF(1, [0.6, 0.4], randomize=False)
    cdf.compute_cdf()

    # 0th index is from 0 - 0.4
    # 1st index is from 0.4 - 0.66667
    # => 0 + (0.4 - 0) * 0.5 = 0.2
    #    0.4 + (0.66667 - 0.4) * 0.5 = 0.533335
    # 0.533335 - 0.2 ~= 0.333
    assert round(cdf.diff_of_two_x(0, .2), 3) == 0.2