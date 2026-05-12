"""Alpaca market data loading for the OU pairs trading bot."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import BotConfig


def load_alpaca_daily_prices(
    data_client: StockHistoricalDataClient,
    config: BotConfig,
) -> pd.DataFrame:
    """Load daily MA/V bars from Alpaca's free/basic IEX feed."""
    start = datetime.fromisoformat(config.data_start).replace(tzinfo=ZoneInfo("America/New_York"))
    end = datetime.now(ZoneInfo("America/New_York"))

    request = StockBarsRequest(
        symbol_or_symbols=config.symbols,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
        start=start,
        end=end,
        adjustment=Adjustment.RAW,
        feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(request).df

    if bars.empty:
        raise ValueError("Alpaca returned no daily bars for the configured symbols.")

    prices = bars["close"].unstack(level=0).dropna().copy()
    missing = sorted(set(config.symbols) - set(prices.columns))
    if missing:
        raise ValueError(f"Missing Alpaca price history for: {missing}")

    prices = prices[config.symbols].copy()
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert(None)

    df = prices.reset_index().rename(
        columns={"timestamp": "Date", config.v_symbol: "V", config.ma_symbol: "MA"}
    )
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    df["V"] = pd.to_numeric(df["V"], errors="coerce")
    df["MA"] = pd.to_numeric(df["MA"], errors="coerce")
    df = df.dropna(subset=["V", "MA"])
    df = df[(df["V"] > 0) & (df["MA"] > 0)].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["V_ret"] = df["V"].pct_change()
    df["MA_ret"] = df["MA"].pct_change()
    df["log_V"] = np.log(df["V"])
    df["log_MA"] = np.log(df["MA"])

    return df

