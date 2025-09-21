import numpy as np
import pandas as pd
from datetime import datetime

class PairsTrade:
    def __init__(self, pair, hedge_ratio, capital_per_leg=10000,
                 entry_threshold=1.0, exit_threshold=0.0, trade_cost_pct=0.0):
        self.pair = pair
        self.hedge_ratio = float(hedge_ratio)
        self.capital_per_leg = float(capital_per_leg)
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.extreme_z_threshold = float(8)
        self.trade_cost_pct = float(trade_cost_pct)  # pct per leg per action

        # position state
        self.position = 0  # 1 = long A, short B ; -1 = short A, long B ; 0 = flat
        self.entry_price_a = None
        self.entry_price_b = None
        self.shares_a = 0.0
        self.shares_b = 0.0
        self.entry_date = None

        # accounting
        self.initial_equity = 2.0 * self.capital_per_leg  # max capital allocated to both legs
        self.deployed_equity = 0.0
        self.realized_pnl = 0.0    # PnL that has been realized (cash)
        self.prev_portfolio_value = self.initial_equity  # start value
        self.prev_unreal = 0.0
        self.daily_returns = []    # list of daily pct returns (period-to-period)
        self.equity_values = []    # list of portfolio values for each date
        self.equity_dates = []     # corresponding dates
        self.pnl = []              # realized trade-level dollar PnL
        self.trades = []           # trade log events

    def mark_to_market_unrealized(self, price_a, price_b):
        """Return unrealized PnL (dollars) for current open position using entry prices."""
        if self.position == 0 or self.entry_price_a is None:
            return 0.0
        if self.position == 1:
            # long A, short B
            unreal = (price_a - self.entry_price_a) * self.shares_a + (-(price_b - self.entry_price_b)) * self.shares_b
        else:
            # short A, long B
            unreal = (-(price_a - self.entry_price_a)) * self.shares_a + (price_b - self.entry_price_b) * self.shares_b
        return unreal

    def _apply_trade_costs(self, notional):
        """Trade cost per action (dollars), so 2x trade cost for round trip for a given notional. """
        return abs(notional) * self.trade_cost_pct

    def update(self, date, price_a, price_b, zscore):
        """
        Called once per calendar/trading date in chronological order.
        1) mark-to-market based on existing positions
        2) record daily return
        3) then process entry/exit signals (entry will affect next day's MTM)
        """

        # 1) compute current unrealized PnL
        current_unreal = self.mark_to_market_unrealized(price_a, price_b)

        # 2) current portfolio value = initial + realized (cash) + unrealized
        current_value = self.initial_equity + self.realized_pnl + current_unreal - self.prev_unreal

        # 3) daily return = pct change vs previous day's portfolio value
        #    (prev_portfolio_value initialized to initial_equity on first day)

        if self.deployed_equity == 0.0:
            daily_ret = 0.0
        else:
            daily_ret = (current_unreal -  self.prev_unreal) / self.deployed_equity


        # Append daily return and equity
        self.daily_returns.append(daily_ret)
        self.equity_values.append(current_value)
        self.equity_dates.append(pd.to_datetime(date))

        # Update prev portfolio value for next day's pct change (do this AFTER append)
        # Note: wont change prev_portfolio_value when we realize PnL (because unreal -> realized doesn't change total)
        self.prev_portfolio_value = current_value
        self.prev_unreal = current_unreal
        # --- Now process signals (entry/exit) AFTER MTM accounted for today ---
        # Entry signals ( enter at today's price; effect shows from next day's MTM)
        if self.position == 0:
            if self.extreme_z_threshold > zscore > self.entry_threshold:
                # ENTER SHORT A, LONG B
                self.position = -1
                self.entry_price_a = price_a
                self.entry_price_b = price_b
                # correct sizing
                beta = abs(self.hedge_ratio)

                max_a_by_budget = self.capital_per_leg / price_a
                max_a_by_b_leg = (self.capital_per_leg / (beta * price_b))

                self.shares_a = min(max_a_by_budget, max_a_by_b_leg)
                self.shares_b = beta * self.shares_a  # absolute shares, signs handled by position
                self.deployed_equity = self.shares_a * self.entry_price_a + self.shares_b * self.entry_price_b

                self.entry_date = pd.to_datetime(date) # for exiting stale trades
                self.trades.append((date, "Enter SHORT", zscore, price_a, price_b, self.shares_a, self.shares_b))

            elif -self.entry_threshold > zscore > -self.extreme_z_threshold:
                # ENTER LONG A, SHORT B
                self.position = 1
                self.entry_price_a = price_a
                self.entry_price_b = price_b
                # correct sizing
                beta = abs(self.hedge_ratio) if abs(self.hedge_ratio) > 0 else 0.0

                max_a_by_budget = self.capital_per_leg / price_a
                max_a_by_b_leg = (self.capital_per_leg / (beta * price_b)) if beta > 0 else np.inf

                self.shares_a = min(max_a_by_budget, max_a_by_b_leg)
                self.shares_b = beta * self.shares_a  # absolute shares, signs handled by position
                self.deployed_equity = self.shares_a * self.entry_price_a + self.shares_b * self.entry_price_b

                self.entry_date = pd.to_datetime(date)
                self.trades.append((date, "Enter LONG", zscore, price_a, price_b, self.shares_a, self.shares_b))

        # Exit signals
        elif (self.position == 1 and zscore >= -self.exit_threshold or current_unreal < -self.deployed_equity*0.15 or
              (pd.to_datetime(date)).day - self.entry_date.day > 90 ): #EXIT LONG
            # apply exit cost
            round_trip_trade_costs = self._apply_trade_costs(self.capital_per_leg * 4) # 2 legs, entry and exit
            net_pnl = current_unreal - round_trip_trade_costs
            # realize it
            self.realized_pnl += net_pnl
            self.pnl.append(net_pnl)
            self.trades.append((date, "Exit LONG", zscore, price_a, price_b, self.shares_a, self.shares_b, net_pnl))
            # clear position
            self.position = 0
            self.entry_price_a = self.entry_price_b = None
            self.shares_a = self.shares_b = self.deployed_equity =  0.0

        elif (self.position == -1 and zscore <= self.exit_threshold or current_unreal < -self.deployed_equity*0.15 or
            (pd.to_datetime(date)).day - self.entry_date.day > 90): #EXIT SHORT
            #apply exit cost
            round_trip_trade_costs = self._apply_trade_costs(self.capital_per_leg * 4)  # 2 legs, entry and exit
            net_pnl = current_unreal - round_trip_trade_costs
            #realize it
            self.realized_pnl += net_pnl
            self.pnl.append(net_pnl)
            self.trades.append((date, "Exit SHORT", zscore, price_a, price_b, self.shares_a, self.shares_b, net_pnl))
            #clear position
            self.position = 0
            self.entry_price_a = self.entry_price_b = None
            self.shares_a = self.shares_b = self.deployed_equity = 0.0

    def results(self):
        # Return daily_returns and equity_curve as pandas Series indexed by dates
        dr = pd.Series(self.daily_returns, index=pd.DatetimeIndex(self.equity_dates))
        eq = pd.Series(self.equity_values, index=pd.DatetimeIndex(self.equity_dates))
        return {
            'pair': (self.pair[0], self.pair[1]),
            'total_pnl': np.sum(self.pnl),
            'num_trades': len(self.pnl),
            'daily_returns': dr,
            'equity_curve': eq,
            'trades': self.trades
        }


def backtest_pairs(zscore_dict, data, hedge_ratios, capital_per_leg=10000,
                   entry=1.0, exit=0.0, trade_cost_pct=0.0):
    """
    zscore_dict: dict mapping pair -> pd.Series of zscores indexed by dates (chronological)
    data: DataFrame of prices with same dates (index)
    hedge_ratios: dict mapping pair -> hedge_ratio
    returns: list of per-pair result dicts (each with daily_returns (Series) and equity_curve (Series))
    """
    results = []
    for pair, zseries in zscore_dict.items():
        trader = PairsTrade(pair, hedge_ratios[pair], capital_per_leg,
                            entry_threshold=entry, exit_threshold=exit, trade_cost_pct=trade_cost_pct)
        # iterate dates in zseries (chronological)
        for date in zseries.index:
            if date not in data.index:
                # If price missing, append a flat day to preserve alignment:
                # Use previous portfolio value to compute zero daily return (no price movement)
                # But simplest: skip date entirely to keep Series consistent with available prices
                continue
            price_a = data.loc[date, pair[0]]
            price_b = data.loc[date, pair[1]]
            zscore = zseries.loc[date]
            trader.update(date, price_a, price_b, zscore)
        results.append(trader.results())
    return results
