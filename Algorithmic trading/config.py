"""Runtime configuration for the OU pairs trading bot."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class BotConfig:
    ma_symbol: str = os.getenv("MA_SYMBOL", "MA")
    v_symbol: str = os.getenv("V_SYMBOL", "V")

    paper: bool = _env_bool("ALPACA_PAPER", True)
    submit_orders: bool = _env_bool("SUBMIT_ORDERS", True)
    data_start: str = os.getenv("DATA_START", "2008-03-19")
    test_start: str = os.getenv("TEST_START", "2020-01-01")
    train_eval_start: str = os.getenv("TRAIN_EVAL_START", "2008-03-19")

    hedge_lookback: int = int(os.getenv("HEDGE_LOOKBACK", "252"))
    ou_lookback: int = int(os.getenv("OU_LOOKBACK", "378"))
    entry_z: float = _env_float("ENTRY_Z", 1.0)
    exit_z: float = _env_float("EXIT_Z", 0.75)
    e_stop: float = _env_float("E_STOP", 3.9)
    min_half_life: float = _env_float("MIN_HALF_LIFE", 1.0)
    max_half_life: float = _env_float("MAX_HALF_LIFE", 60.0)
    min_hold_days: int = int(os.getenv("MIN_HOLD_DAYS", "1"))
    cooldown_days: int = int(os.getenv("COOLDOWN_DAYS", "1"))

    regime_lookback: int = int(os.getenv("REGIME_LOOKBACK", "126"))
    beta_stability_lookback: int = int(os.getenv("BETA_STABILITY_LOOKBACK", "20"))
    min_corr_quantile: float = _env_float("MIN_CORR_QUANTILE", 0.75)
    max_beta_change_quantile: float = _env_float("MAX_BETA_CHANGE_QUANTILE", 0.90)
    max_ou_sigma_quantile: float = _env_float("MAX_OU_SIGMA_QUANTILE", 0.90)

    use_correlation_filter: bool = _env_bool("USE_CORRELATION_FILTER", True)
    use_beta_stability_filter: bool = _env_bool("USE_BETA_STABILITY_FILTER", True)
    use_ou_sigma_filter: bool = _env_bool("USE_OU_SIGMA_FILTER", True)

    max_gross_exposure: float = _env_float("MAX_GROSS_EXPOSURE", 100000.0)
    round_to_whole_shares: bool = _env_bool("ROUND_TO_WHOLE_SHARES", True)
    min_trade_notional: float = _env_float("MIN_TRADE_NOTIONAL", 10.0)

    @property
    def symbols(self) -> list[str]:
        return [self.v_symbol, self.ma_symbol]

    @property
    def paper_endpoint(self) -> str:
        return "https://paper-api.alpaca.markets/v2"
