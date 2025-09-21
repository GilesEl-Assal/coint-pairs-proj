import pandas as pd
import numpy as np
import yfinance as yf
from itertools import combinations
from datetime import datetime

def data_download(tickers, start, end, max_missing_ratio=0.05):
    df = yf.download(tickers, start, end, auto_adjust=False)["Adj Close"]
    df.index = pd.to_datetime(df.index)

    null_ratio = df.isnull().mean()
    to_drop = null_ratio[null_ratio > max_missing_ratio].index.tolist()
    if to_drop:
        print(f"Dropping tickers due to excess missing data: {to_drop}")
        df = df.drop(columns=to_drop)

    df = df.ffill().dropna()
    return df if not df.empty else None

def update_day():
    return datetime.today().date()
