"""OU strategy implementation copied from the validated notebook logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BotConfig


def estimate_ou_params(spread_series):
    """Estimate discrete OU/AR(1) parameters from a trailing spread window."""
    x = np.asarray(spread_series, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 20 or np.std(x[:-1]) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x_lag = x[:-1]
    x_next = x[1:]
    phi, intercept = np.polyfit(x_lag, x_next, 1)

    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    theta = -np.log(phi)
    mu = intercept / (1 - phi)
    residuals = x_next - (intercept + phi * x_lag)
    residual_sigma = np.std(residuals, ddof=1)
    stationary_sigma = residual_sigma / np.sqrt(1 - phi**2)
    half_life = np.log(2) / theta if theta > 0 else np.nan

    return mu, phi, theta, stationary_sigma, half_life


def add_ou_features(price_df: pd.DataFrame, config: BotConfig) -> pd.DataFrame:
    features = price_df.copy().reset_index(drop=True)

    for col in ["beta", "spread", "ou_mu", "ou_phi", "ou_theta", "ou_sigma", "half_life", "ou_z"]:
        features[col] = np.nan

    log_v = features["log_V"].to_numpy()
    log_ma = features["log_MA"].to_numpy()
    start = max(config.hedge_lookback, config.ou_lookback + 2)

    for i in range(start, len(features)):
        hedge_start = i - config.hedge_lookback
        beta = np.polyfit(log_v[hedge_start:i], log_ma[hedge_start:i], 1)[0]
        features.loc[i, "beta"] = beta

        ou_start = i - config.ou_lookback
        spread_history = log_ma[ou_start:i] - beta * log_v[ou_start:i]
        ou_mu, ou_phi, ou_theta, ou_sigma, half_life = estimate_ou_params(spread_history)
        current_spread = log_ma[i] - beta * log_v[i]

        features.loc[i, "spread"] = current_spread
        features.loc[i, "ou_mu"] = ou_mu
        features.loc[i, "ou_phi"] = ou_phi
        features.loc[i, "ou_theta"] = ou_theta
        features.loc[i, "ou_sigma"] = ou_sigma
        features.loc[i, "half_life"] = half_life

        if np.isfinite(ou_sigma) and ou_sigma > 0:
            features.loc[i, "ou_z"] = (current_spread - ou_mu) / ou_sigma

    features["return_corr"] = features["V_ret"].rolling(config.regime_lookback).corr(features["MA_ret"])
    features["beta_change"] = features["beta"].diff().abs()
    features["beta_change_avg"] = features["beta_change"].rolling(config.beta_stability_lookback).mean()

    return features


def calibrate_regime_thresholds(feature_df: pd.DataFrame, config: BotConfig) -> dict[str, float]:
    training_features = feature_df[
        (feature_df["Date"] >= "2000-01-01") & (feature_df["Date"] < config.test_start)
    ].copy()

    return {
        "min_corr": training_features["return_corr"].quantile(config.min_corr_quantile),
        "max_beta_change": training_features["beta_change_avg"].quantile(config.max_beta_change_quantile),
        "max_ou_sigma": training_features["ou_sigma"].quantile(config.max_ou_sigma_quantile),
    }


def run_ou_strategy(
    feature_df: pd.DataFrame,
    label: str,
    evaluate_from: str,
    evaluate_to: str | None,
    regime_thresholds: dict[str, float],
    config: BotConfig,
) -> dict[str, object]:
    result = feature_df.copy().reset_index(drop=True)

    result["corr_ok"] = True
    result["beta_stable"] = True
    result["ou_sigma_ok"] = True

    if config.use_correlation_filter:
        result["corr_ok"] = result["return_corr"] >= regime_thresholds["min_corr"]
    if config.use_beta_stability_filter:
        result["beta_stable"] = result["beta_change_avg"] <= regime_thresholds["max_beta_change"]
    if config.use_ou_sigma_filter:
        result["ou_sigma_ok"] = result["ou_sigma"] <= regime_thresholds["max_ou_sigma"]

    result["half_life_ok"] = result["half_life"].between(config.min_half_life, config.max_half_life)
    result["regime_ok"] = result[["corr_ok", "beta_stable", "ou_sigma_ok", "half_life_ok"]].all(axis=1)
    result["position"] = 0.0

    start_matches = result.index[result["Date"] >= evaluate_from]
    if len(start_matches) == 0:
        raise ValueError(f"No data available at or after evaluate_from={evaluate_from!r}.")
    start_idx = int(start_matches[0])

    if evaluate_to is None:
        end_idx = len(result)
    else:
        end_candidates = result.index[result["Date"] >= evaluate_to]
        end_idx = int(end_candidates[0]) if len(end_candidates) else len(result)

    current_pos = 0
    entry_idx = None
    cooldown_remaining = 0

    for i in range(start_idx, end_idx):
        z = result.loc[i, "ou_z"]
        regime_ok = bool(result.loc[i, "regime_ok"])

        if pd.isna(z):
            result.loc[i, "position"] = current_pos
            continue

        if current_pos == 0:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            elif regime_ok:
                if z > config.entry_z:
                    current_pos = -1
                    entry_idx = i
                elif z < -config.entry_z:
                    current_pos = 1
                    entry_idx = i

        elif current_pos == 1:
            holding_days = i - entry_idx if entry_idx is not None else 0
            if z < -config.e_stop or not regime_ok or (
                holding_days >= config.min_hold_days and z > -config.exit_z
            ):
                current_pos = 0
                entry_idx = None
                cooldown_remaining = config.cooldown_days

        elif current_pos == -1:
            holding_days = i - entry_idx if entry_idx is not None else 0
            if z > config.e_stop or not regime_ok or (
                holding_days >= config.min_hold_days and z < config.exit_z
            ):
                current_pos = 0
                entry_idx = None
                cooldown_remaining = config.cooldown_days

        result.loc[i, "position"] = current_pos

    result["beta_ffill"] = result["beta"].ffill()
    result["spread_ret"] = result["MA_ret"] - result["beta_ffill"] * result["V_ret"]
    result["position_lag"] = result["position"].shift(1).fillna(0)
    result["strategy_ret"] = (result["position_lag"] * result["spread_ret"]).fillna(0)

    result = result.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    result["equity_curve"] = (1 + result["strategy_ret"]).cumprod()
    result["drawdown"] = result["equity_curve"] / result["equity_curve"].cummax() - 1

    mean_ret = result["strategy_ret"].mean()
    std_ret = result["strategy_ret"].std()
    sharpe = np.nan if std_ret == 0 else np.sqrt(252) * mean_ret / std_ret

    return {
        "label": label,
        "start_date": result["Date"].min(),
        "end_date": result["Date"].max(),
        "sharpe": sharpe,
        "total_return": result["equity_curve"].iloc[-1] - 1,
        "max_drawdown": result["drawdown"].min(),
        "num_trades": int((result["position"].diff().abs() > 0).sum()),
        "exposure": result["position"].ne(0).mean(),
        "regime_ok_rate": result["regime_ok"].mean(),
        "final_equity": result["equity_curve"].iloc[-1],
        "backtest_df": result,
    }


def latest_ou_signal(price_df: pd.DataFrame, config: BotConfig) -> tuple[pd.Series, dict[str, float], dict[str, object]]:
    feature_df = add_ou_features(price_df, config)
    regime_thresholds = calibrate_regime_thresholds(feature_df, config)
    validation_result = run_ou_strategy(
        feature_df=feature_df,
        label="OU validation",
        evaluate_from=config.test_start,
        evaluate_to=None,
        regime_thresholds=regime_thresholds,
        config=config,
    )
    validation_df = validation_result["backtest_df"]
    return validation_df.iloc[-1], regime_thresholds, validation_result

