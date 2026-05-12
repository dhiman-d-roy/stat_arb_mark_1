"""Run the OU pairs trading bot once.

Schedule this script with cron or another scheduler for deployment.
By default the bot submits orders to Alpaca paper trading.
Set SUBMIT_ORDERS=false to print the intended orders without submitting them.
"""

from __future__ import annotations

from config import BotConfig
from data import load_alpaca_daily_prices
from execution import build_order_plan, create_clients, execute_order_plan
from strategy import latest_ou_signal


def main() -> None:
    config = BotConfig()
    if not config.paper:
        raise ValueError("This bot is configured for paper trading only. Set ALPACA_PAPER=true.")

    print(f"Alpaca paper trading endpoint: {config.paper_endpoint}")
    print(f"Order submission enabled: {config.submit_orders}")

    trade_client, data_client = create_clients(config)
    account = trade_client.get_account()
    print(f"Account status: {account.status}")
    print(f"Buying power: {account.buying_power}")

    price_df = load_alpaca_daily_prices(data_client, config)
    latest_signal, regime_thresholds, validation_result = latest_ou_signal(price_df, config)

    print("\nLatest OU signal")
    print(latest_signal[["Date", "V", "MA", "beta", "ou_z", "regime_ok", "position"]].to_string())

    print("\nRegime thresholds")
    for key, value in regime_thresholds.items():
        print(f"{key}: {value:.6f}")

    print("\nValidation summary")
    for key in ["start_date", "end_date", "sharpe", "total_return", "max_drawdown", "num_trades", "exposure"]:
        print(f"{key}: {validation_result[key]}")

    order_plan = build_order_plan(trade_client, latest_signal, config)
    print("\nOrder plan")
    print(order_plan.to_string(index=False))

    execute_order_plan(trade_client, order_plan, config)


if __name__ == "__main__":
    main()
