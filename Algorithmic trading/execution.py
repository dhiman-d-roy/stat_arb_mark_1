"""Paper-trading execution helpers for Alpaca."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config import BotConfig


def load_credentials() -> tuple[str, str]:
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if api_key and secret_key:
        return api_key, secret_key

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from alpaca_keys import AlpacaKeys
    except ImportError as exc:
        raise ValueError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY, "
            "or keep alpaca_keys.py available at the project root."
        ) from exc

    keys = AlpacaKeys()
    api_key = keys.api_key
    secret_key = keys.secret_key
    if not api_key or not secret_key:
        raise ValueError("Missing Alpaca API keys.")
    return api_key, secret_key


def create_clients(config: BotConfig) -> tuple[TradingClient, StockHistoricalDataClient]:
    api_key, secret_key = load_credentials()
    trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=config.paper)
    data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    return trade_client, data_client


def target_notional_from_position(position: float, beta: float, gross_exposure: float, config: BotConfig) -> dict[str, float]:
    if position == 0 or not np.isfinite(beta):
        return {config.ma_symbol: 0.0, config.v_symbol: 0.0}

    ma_notional = position * gross_exposure / (1 + abs(beta))
    v_notional = -position * beta * gross_exposure / (1 + abs(beta))
    return {config.ma_symbol: ma_notional, config.v_symbol: v_notional}


def notional_to_qty(notional: float, price: float, config: BotConfig) -> float:
    raw_qty = notional / price
    if not config.round_to_whole_shares:
        return round(raw_qty, 6)
    return float(np.sign(raw_qty) * np.floor(abs(raw_qty)))


def get_current_qty(trade_client: TradingClient, symbol: str) -> float:
    try:
        position = trade_client.get_open_position(symbol)
        return float(position.qty)
    except Exception:
        return 0.0


def build_order_plan(trade_client: TradingClient, latest_signal: pd.Series, config: BotConfig) -> pd.DataFrame:
    latest_prices = {
        config.ma_symbol: float(latest_signal["MA"]),
        config.v_symbol: float(latest_signal["V"]),
    }
    target_notional = target_notional_from_position(
        position=float(latest_signal["position"]),
        beta=float(latest_signal["beta"]),
        gross_exposure=config.max_gross_exposure,
        config=config,
    )

    order_plan = []
    for symbol in [config.ma_symbol, config.v_symbol]:
        current_qty = get_current_qty(trade_client, symbol)
        target_qty = notional_to_qty(target_notional[symbol], latest_prices[symbol], config)
        trade_qty = target_qty - current_qty
        trade_notional = trade_qty * latest_prices[symbol]
        order_plan.append(
            {
                "symbol": symbol,
                "latest_price": latest_prices[symbol],
                "current_qty": current_qty,
                "target_notional": target_notional[symbol],
                "target_qty": target_qty,
                "trade_qty": trade_qty,
                "trade_notional": trade_notional,
            }
        )

    return pd.DataFrame(order_plan)


def submit_market_order(
    trade_client: TradingClient,
    symbol: str,
    trade_qty: float,
    latest_price: float,
    config: BotConfig,
):
    trade_notional = abs(trade_qty) * latest_price
    if abs(trade_qty) == 0 or trade_notional < config.min_trade_notional:
        print(f"Skipping {symbol}: trade is below threshold.")
        return None

    side = OrderSide.BUY if trade_qty > 0 else OrderSide.SELL
    qty = abs(trade_qty)
    if config.round_to_whole_shares:
        qty = int(qty)

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )

    if not config.submit_orders:
        print(f"[DRY RUN] Would submit paper {side.value} order for {qty} share(s) of {symbol}.")
        return None

    result = trade_client.submit_order(order)
    print(f"Submitted paper {side.value} order for {qty} share(s) of {symbol}.")
    return result


def execute_order_plan(trade_client: TradingClient, order_plan: pd.DataFrame, config: BotConfig) -> list[object]:
    submitted_orders = []
    for row in order_plan.itertuples(index=False):
        submitted_orders.append(
            submit_market_order(
                trade_client=trade_client,
                symbol=row.symbol,
                trade_qty=row.trade_qty,
                latest_price=row.latest_price,
                config=config,
            )
        )
    return submitted_orders

