from fma.mmar.trading_time_cdf import TradingTimeCDF

def test_diff_at_index_with_pct():
    tradingTime = TradingTimeCDF(1, [0.6, 0.4], randomize=False)
    tradingTime.create_trading_time_cdf()

    assert round(tradingTime.cdf(.2) - tradingTime.cdf(0), 3) == 0.2