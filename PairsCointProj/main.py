import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from coint_corrs_utils import (
    find_coint_pairs,
    vol_check,
    hedge_ratio_dict,
    calc_spread,
    check_stationarity,
    rolling_zscore,
    rolling_correlations,
    rolling_coint_test,
    corr_summary,
    recalc_flipped_hedge_ratios
)
from optimization_evaluation_utils import (
    calculate_sharpe_ratio,
    diagnose_pair
)
from backtest import backtest_pairs
from data_utils import (
    data_download,
    update_day
)
from datetime import datetime
pd.set_option("display.max_columns", None)

# === CONFIGURATION ===

current_day = update_day()
start_date = "2021-08-17"
end_date = str(current_day)

initial_equity = 20000 #Pick arbitrarily

ticker_list = [
                #Retail and Consumer Staples
                "WMT", "TGT", "HD", "LOW", "COST", "BJ", "KO", "PEP", "PG", "CL", "MDLZ", "KHC","KO","PEP"
                #Utilities
                #"DUK", "D", "NEE", "SO", "AEP", "EXC", "ED", "PEG", "XEL", "CMS"

                ]

# === STEP 1: Data Download ===
data = data_download(ticker_list, start_date, end_date)
if data is None:
    raise ValueError("No valid data after cleaning.")

# Train test split for parameter optimization

split_date = data.index[int(len(data) * 0.7)]

train_data = data.loc[:split_date]
test_data  = data.loc[split_date+pd.Timedelta(days=1):]


# Step 2: Find cointegrated pairs only on train_data
coint_pairs_train = find_coint_pairs(train_data)
ordered_pairs_train = vol_check(coint_pairs_train, train_data)


# Step 3: Calculate hedge ratios only on train_data
hedge_ratios_train = hedge_ratio_dict(ordered_pairs_train, train_data)
flipped_ratios_train = recalc_flipped_hedge_ratios(hedge_ratios_train, train_data)


# Step 4: Calculate spread and zscores only on full dataset
spread_full = calc_spread(flipped_ratios_train, data)

''' Checks the zscored spread dictionary to find outliers '''
#for pair in spread_full:
#    z, sigma, spikes = diagnose_pair(spread_full[pair], train_data, pair, lookback=21)








# Optimize entry and exit signals, as well as different lookback vales in the zscore dictionary

lookback_values =[21, 60, 90, 120, 150, 300]
entry_values = [1.0, 1.25, 1.5, 1.75, 2.0]
exit_values  = [0.0, 0.125, 0.25, 0.375, 0.5]

best_params = {}
for pair in flipped_ratios_train.keys():
    best_sharpe_pair = -np.inf
    best_param_pair = None

    for lookback in lookback_values:
        # Compute z-score for each lookback
        zscore_full = rolling_zscore(spread_full, lookback)
        zscore_train = zscore_full.loc[train_data.index]

        zscore_pair = zscore_train[pair].dropna()
        data_pair = train_data[[pair[0], pair[1]]]



        for entry in entry_values:
            for exit in exit_values:
                # Backtest one pair at a time
                results = backtest_pairs({pair: zscore_pair}, data_pair, {pair: flipped_ratios_train[pair]},
                                    initial_equity, entry=entry, exit=exit,trade_cost_pct=0.005)
                daily_returns = results[0]['daily_returns']

                #mean_ret = np.mean(daily_returns)
                #std_ret = np.std(daily_returns, ddof=1)
                #max_ret = np.max(daily_returns)        Prints stats from training data trades for sanity checks/debugging optimization
                #min_ret = np.min(daily_returns)

                sharpe = calculate_sharpe_ratio(daily_returns)

                #print(
                #f"{pair} Entry={entry}, Exit={exit} -> Trades: {results[0]['num_trades']}, Mean: {mean_ret:.5f}, Std: {std_ret:.5f}, "
                #f"Max: {max_ret:.5f}, Min: {min_ret:.5f}, Sharpe: {sharpe:.2f}")

                if sharpe > best_sharpe_pair:
                    best_sharpe_pair = sharpe
                    best_param_pair = (lookback, entry, exit)

    if best_param_pair is not None:
        best_params[pair] = best_param_pair
        print(
            f"Best params for {pair}: Lookback={best_param_pair[0]}, Entry={best_param_pair[1]}, Exit={best_param_pair[2]}, "
            f"Sharpe={best_sharpe_pair:.2f}")
    else:
        print(f"No valid params found for {pair}")


# Now use the best parameters for each pair to trade in the test window

for pair, (best_lookback, best_entry, best_exit) in best_params.items():
    #recompute z-score with the chosen lookback
    zscore_full = rolling_zscore(spread_full,best_lookback)
    zscore_test = zscore_full.loc[test_data.index]

    zscore_pair = zscore_test[pair]
    data_pair = test_data[[pair[0], pair[1]]]
    results = backtest_pairs({pair: zscore_pair}, data_pair, {pair: flipped_ratios_train[pair]},
                             initial_equity, entry=best_entry, exit=best_exit,trade_cost_pct=0.005)
    daily_returns = results[0]['daily_returns']
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns, ddof=1)
    max_ret = np.max(daily_returns)
    min_ret = np.min(daily_returns)

    sharpe = calculate_sharpe_ratio(daily_returns)
    #Evaluation metrics: Sharpe
    print(
        f"TEST RESULTS: {pair} Lookback={best_lookback}, Entry={best_params[pair][1]}, Exit={best_params[pair][2]} "
        f"-> Trades: {results[0]['num_trades']}, Mean: {mean_ret:.5f}, Std: {std_ret:.5f}, "
        f"Max: {max_ret:.5f}, Min: {min_ret:.5f}, Sharpe: {sharpe:.2f}")
    #Daily Return Graph
    for res in results:
        pair_name = res['pair']
        return_graph = res['daily_returns']

        plt.figure(figsize=(12, 6))
        return_graph.plot(label=f"Returns: {pair_name[0]}-{pair_name[1]}")
        plt.title(f"Return graph for {pair_name[0]} vs {pair_name[1]}")
        plt.xlabel("Date")
        plt.ylabel("Daily Returns ")
        plt.legend()
        plt.show()
    #Equity Curves
    for res in results:
        pair_name = res['pair']
        equity_curve = res['equity_curve']

        plt.figure(figsize=(12, 6))
        equity_curve.plot(label=f"Equity: {pair_name[0]}-{pair_name[1]}")
        plt.title(f"Equity Curve for {pair_name[0]} vs {pair_name[1]}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.show()

''' LOG OF EVERY TRADE (ENTRY + EXIT DATES, NUMBER OF SHARES, AND NET PNL)'''

for pair, (best_lookback, best_entry, best_exit) in best_params.items():
    #recompute z-score with the chosen lookback
    zscore_full = rolling_zscore(spread_full, best_lookback)
    zscore_test = zscore_full.loc[test_data.index]

    zscore_pair = zscore_test[pair]
    data_pair = test_data[[pair[0], pair[1]]]
    results = backtest_pairs({pair: zscore_pair}, data_pair, {pair: flipped_ratios_train[pair]},
                             initial_equity, entry=best_entry, exit=best_exit,trade_cost_pct=0.0005)

    print(f"Trades for pair {pair}:")
    for trade in results[0]["trades"]:
        # unpack trade tuple depending on your structure
        if "Enter" in trade[1]:
            date, action, zscore, price_a, price_b, shares_a, shares_b = trade
            print(f"{date.date()} | {action:10} | z={zscore:.2f} | "
                f"Price A= {price_a:.2f}, Price B= {price_b:.2f} | Shares A= {shares_a:.2f}, Shares B = {shares_b:.2f}")
        else:
            date, action, zscore, price_a, price_b, shares_a, shares_b, net_pnl = trade
            print(f"{date.date()} | {action:10} | z={zscore:.2f} | "
                f"Price A= {price_a:.2f}, Price B= {price_b:.2f} | Shares A= {shares_a:.2f}, Shares B = {shares_b:.2f}"
                f" | Net Pnl = {net_pnl:.2f}")






