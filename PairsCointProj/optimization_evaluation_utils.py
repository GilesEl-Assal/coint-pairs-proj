import numpy as np
import pandas as pd
from coint_corrs_utils import rolling_zscore




def calculate_sharpe_ratio(returns, risk_free_rate=0.0,periods_per_year=252):
    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)
    std_dev = np.std(excess_returns, ddof=1)
    if std_dev == 0:
        return np.nan
    sharpe_ratio = (mean_return / std_dev) * np.sqrt(periods_per_year)
    return sharpe_ratio



def diagnose_pair(spread_series, prices_df, pair, lookback=21, ddof=0, z_threshold=6):
    # rolling stats (no dropna so index stays intact)
    spread_series = spread_series.copy()
    spread_series.index = pd.to_datetime(spread_series.index)

    prices_df = prices_df.copy()
    prices_df.index = pd.to_datetime(prices_df.index)

    prices_df = prices_df.reindex(prices_df.index.union(spread_series.index))
    r = spread_series.rolling(window=lookback)
    m = r.mean().shift(1)
    sigma = r.std(ddof).shift(1)
    z = (spread_series - m) / sigma

    print(f"=== SIGMA SUMMARY {pair} ===")
    print(sigma.describe())
    print("count zeros in sigma:", int((sigma == 0).sum()))
    print("count tiny sigma (<1e-6):", int((sigma.abs() < 1e-6).sum()))
    print("count NaNs in sigma:", int(sigma.isna().sum()))

    # Find spike dates
    spikes = z[ z.abs() > z_threshold ].dropna()
    print("\nTotal z > {} spikes: {}".format(z_threshold, len(spikes)))
    if len(spikes) == 0:
        print("No extreme z spikes found at threshold.")
    else:
        print(f"First {len(spikes)} spike dates (date, z, sigma, spread, priceA, priceB):")
        for d, val in list(spikes.items())[:50]:
            print(d.date(), round(val, 3),
                  " sigma=", round(sigma.loc[d],6),
                  " spread=", round(spread_series.loc[d],6),
                  f" {pair[0]}={prices_df.loc[d, pair[0]]:.3f}",
                  f" {pair[1]}={prices_df.loc[d, pair[1]]:.3f}")

    return z, sigma, spikes

# Example usage:
# spread = spread_full[(pair)]  # pandas Series for that pair
# z, sigma, spikes = diagnose_pair(spread, data, pair, lookback=21)