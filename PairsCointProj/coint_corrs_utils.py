from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import numpy as np
import pandas as pd

# === INITIAL  ===

def rolling_correlations(data, window_size=21):
    log_data = np.log(data)
    pairs = list(combinations(log_data.columns, 2))
    rolling_corr_dict = {}

    for pair in pairs:
        rolling_corr = log_data[pair[0]].rolling(window=window_size).corr(log_data[pair[1]])
        rolling_corr_dict[f"{pair[0]}_{pair[1]}"] = rolling_corr.dropna()

    return pd.DataFrame(rolling_corr_dict)

def corr_summary(data, window_size=21):
    rolling_corr_dict = rolling_correlations(data, window_size=window_size)
    cor_stats_dict = {}

    for pair, series in rolling_corr_dict.items():
        stats = {
            'max': series.max(),
            'min': series.min(),
            'mean': series.mean(),
            'std': series.std()
        }
        cor_stats_dict[pair] = stats
    return pd.DataFrame(cor_stats_dict)

# Find cointegrated pairs adf. First result is good pair list, second is pairs with their pvals
def find_coint_pairs(data, alpha=0.05):
    rows = []
    tickers = data.columns
    pairs = list(combinations(tickers, 2))
    result= []
    for pair in pairs:
        pval = coint(data[pair[0]], data[pair[1]])[1]
        if pval < alpha:
            result.append(pair)
            rows.append({
                "Pair": pair,
                "p-value": pval,
            })
    # rejected, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    # significant_pairs = [pair for pair, is_sig in zip(pairs, rejected) if is_sig]
    return result, pd.DataFrame(rows)



# Rolling cointegration window testing
def rolling_coint_test(data, window=252, step=21):
    results = []
    for start in range(0, len(data) - window + 1, step):
        end = start + window
        window_data = data.iloc[start:end]
        pairs, _ = find_coint_pairs(window_data)
        results.append({"start": window_data.index[0], "end": window_data.index[-1], 'pairs': pairs})
    return results

# Ensure more volatile stock is listed first
def vol_check(pairs, data):
    voldata = data.pct_change().std().dropna()
    orderedpairs = []
    for pair in pairs[0]:
        if voldata[pair[0]] > voldata[pair[1]]:
            orderedpairs.append((pair[0], pair[1]))
        else:
            orderedpairs.append((pair[1], pair[0]))
    return orderedpairs

# Compute hedge ratios via OLS regression
def hedge_ratio_dict(pairs, data):
    hedge_ratios = {}
    for pair in pairs:
        x = sm.add_constant(data.loc[:, pair[1]])
        model = sm.OLS(data.loc[:, pair[0]], x, missing="raise").fit()
        hedge_ratios[pair] = model.params[pair[1]]
    return hedge_ratios


def recalc_flipped_hedge_ratios(hedge_ratios, data):
    updated_hedge_ratios = {}

    for pair, ratio in hedge_ratios.items():
        if ratio >= 0:
            # Keep pair and ratio as is
            updated_hedge_ratios[pair] = ratio
        else:
            # Flip the pair order
            flipped_pair = (pair[1], pair[0])
            x = sm.add_constant(data[flipped_pair[1]])
            y = data[flipped_pair[0]]
            model = sm.OLS(y, x, missing="raise").fit()

            new_ratio = model.params[flipped_pair[1]]
            if new_ratio >= 0:
                updated_hedge_ratios[flipped_pair] = new_ratio

    return updated_hedge_ratios

# Calculate spread using hedge ratios
def calc_spread(hedge_ratios, data):
    spread_dict = {}
    for pair in hedge_ratios:
        spread_dict[pair] = data.loc[:, pair[0]] - data.loc[:, pair[1]] * hedge_ratios[pair]
    return pd.DataFrame(spread_dict)

# # Check stationarity of spreads

def check_stationarity(spreads):
    for pair, spread in spreads.items():
        adf_result = adfuller(spread.dropna())
        if adf_result[1] > 0.05:
            print(f"ADF test p-value for {pair} is: {adf_result[1]}")
    return None


# Z-score normalization
def rolling_zscore(series, window):
    r = series.rolling(window=window)
    m = r.mean().shift(1)
    sigma = r.std(ddof=0).shift(1)
    z = (series - m) / sigma

    return z



