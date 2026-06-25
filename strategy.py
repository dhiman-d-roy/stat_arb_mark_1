import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


class PairsTradingStrategy:
    def __init__(
        self,
        pair_data,
        tuned_params,
        active_pairs=None,
        regime_lookback=252,
        coint_pvalue_threshold=0.1,
        min_half_life=2,
        max_half_life=63,
        target_spread_vol=0.01,
        max_pair_position=1.0,
        max_gross_exposure=4.0,
        transaction_cost_bps=1.0,
    ):
        self.pair_data = {
            self._pair_tuple(pair): df.copy().sort_values("Date").reset_index(drop=True)
            for pair, df in pair_data.items()
        }
        self.tuned_params = self._normalize_tuned_params(tuned_params)
        self.active_pairs = (
            [self._pair_tuple(pair) for pair in active_pairs]
            if active_pairs is not None
            else list(self.tuned_params.keys())
        )

        self.regime_lookback = regime_lookback
        self.coint_pvalue_threshold = coint_pvalue_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.target_spread_vol = target_spread_vol
        self.max_pair_position = max_pair_position
        self.max_gross_exposure = max_gross_exposure
        self.transaction_cost_bps = transaction_cost_bps

        self.state = {pair: self._empty_pair_state() for pair in self.active_pairs}

    def _empty_pair_state(self):
        return {
            "theta": np.array([0.0, 1.0]),
            "p_cov": np.eye(2),
            "last_date": None,
            "spread_history": [],
            "spread_change_history": [],
            "raw_signal": 0,
            "target_position": 0.0,
            "last_target_position": 0.0,
            "alpha": np.nan,
            "beta": np.nan,
            "spread": np.nan,
            "z": np.nan,
        }

    def _pair_tuple(self, pair):
        if isinstance(pair, tuple):
            return pair
        if isinstance(pair, str):
            left, right = pair.split("/")
            return left, right
        raise TypeError("pair must be a tuple like ('XOM', 'CVX') or a string like 'XOM/CVX'.")

    def _pair_name(self, pair):
        return f"{pair[0]}/{pair[1]}"

    def _normalize_tuned_params(self, tuned_params):
        if isinstance(tuned_params, pd.DataFrame):
            records = tuned_params.to_dict("records")
        elif isinstance(tuned_params, list):
            records = tuned_params
        elif isinstance(tuned_params, dict) and "pair" in tuned_params:
            records = [tuned_params]
        elif isinstance(tuned_params, dict):
            return {self._pair_tuple(pair): params.copy() for pair, params in tuned_params.items()}
        else:
            raise TypeError("tuned_params must be a dict, list of dicts, or DataFrame.")

        normalized = {}
        for params in records:
            pair = self._pair_tuple(params["pair"])
            normalized[pair] = params.copy()
        return normalized

    def set_active_pairs(self, active_pairs):
        self.active_pairs = [self._pair_tuple(pair) for pair in active_pairs]
        for pair in self.active_pairs:
            self.state.setdefault(pair, self._empty_pair_state())

    def update_tuned_params(self, tuned_params):
        self.tuned_params.update(self._normalize_tuned_params(tuned_params))

    def _params(self, pair):
        if pair not in self.tuned_params:
            raise KeyError(f"No tuned parameters found for {self._pair_name(pair)}.")
        return self.tuned_params[pair]

    def _rows_to_update(self, pair, date):
        df = self.pair_data[pair]
        date = pd.to_datetime(date)
        mask = df["Date"] <= date

        last_date = self.state[pair]["last_date"]
        if last_date is not None:
            mask &= df["Date"] > last_date

        return df.loc[mask].copy()

    def _kalman_step(self, pair, row):
        params = self._params(pair)
        state = self.state[pair]
        y_ticker, x_ticker = pair

        y_value = float(row[f"log_{y_ticker}"])
        x_value = float(row[f"log_{x_ticker}"])

        theta = state["theta"]
        p_cov = state["p_cov"]
        q_cov = np.array([
            [params["q_alpha"], 0.0],
            [0.0, params["q_beta"]],
        ])
        r_var = params["r"]

        h_obs = np.array([1.0, x_value])
        theta_pred = theta
        p_pred = p_cov + q_cov

        y_pred = h_obs @ theta_pred
        innovation = y_value - y_pred
        innovation_var = h_obs @ p_pred @ h_obs.T + r_var

        kalman_gain = p_pred @ h_obs.T / innovation_var
        theta = theta_pred + kalman_gain * innovation
        p_cov = p_pred - np.outer(kalman_gain, h_obs) @ p_pred

        spread = y_value - theta[0] - theta[1] * x_value
        previous_spread = state["spread"]

        state["theta"] = theta
        state["p_cov"] = p_cov
        state["last_date"] = pd.to_datetime(row["Date"])
        state["alpha"] = theta[0]
        state["beta"] = theta[1]
        state["spread"] = spread
        state["spread_history"].append(spread)

        if not pd.isna(previous_spread):
            state["spread_change_history"].append(spread - previous_spread)

        state["z"] = self._compute_z(pair)

    def update_kalman_hedge_ratio(self, pair, date):
        pair = self._pair_tuple(pair)
        rows = self._rows_to_update(pair, date)

        for _, row in rows.iterrows():
            self._kalman_step(pair, row)

        return self.state[pair]

    def _compute_z(self, pair):
        params = self._params(pair)
        lookback = int(params["z_lookback"])
        spread_history = self.state[pair]["spread_history"]

        if len(spread_history) <= lookback:
            return np.nan

        trailing = pd.Series(spread_history[-lookback - 1:-1])
        spread_mean = trailing.mean()
        spread_std = trailing.std()

        if spread_std == 0 or pd.isna(spread_std):
            return np.nan

        return (spread_history[-1] - spread_mean) / spread_std

    def cointegration_gate(self, pair, date):
        pair = self._pair_tuple(pair)
        y_ticker, x_ticker = pair
        df = self.pair_data[pair]
        date = pd.to_datetime(date)
        window = df[df["Date"] <= date].tail(self.regime_lookback).copy()

        if len(window) < 30:
            return {
                "passed": False,
                "reason": "Insufficient data",
                "adf_pvalue": np.nan,
                "adf_stat": np.nan,
            }

        y = window[f"log_{y_ticker}"]
        x = window[f"log_{x_ticker}"]
        model = sm.OLS(y, sm.add_constant(x)).fit()

        residual = y - model.params.iloc[0] - model.params.iloc[1] * x
        adf_result = adfuller(residual.dropna())
        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]

        return {
            "passed": adf_pvalue < self.coint_pvalue_threshold,
            "alpha": model.params.iloc[0],
            "beta": model.params.iloc[1],
            "residual": residual,
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "reason": "Passed" if adf_pvalue < self.coint_pvalue_threshold else "ADF p-value too high",
        }

    def mean_reversion_gate(self, residual):
        residual = residual.dropna()

        if len(residual) < 30:
            return {
                "passed": False,
                "phi": np.nan,
                "half_life": np.nan,
                "reason": "Insufficient data",
            }

        x = residual.iloc[:-1]
        y = residual.iloc[1:]
        phi, _ = np.polyfit(x, y, 1)

        if phi <= 0 or phi >= 1:
            return {
                "passed": False,
                "phi": phi,
                "half_life": np.nan,
                "reason": "Not mean-reverting",
            }

        theta = -np.log(phi)
        half_life = np.log(2) / theta
        passed = self.min_half_life <= half_life <= self.max_half_life

        return {
            "passed": passed,
            "phi": phi,
            "theta": theta,
            "half_life": half_life,
            "reason": "Passed" if passed else "Bad half-life",
        }

    def regime_filter(self, pair, date):
        cointegration = self.cointegration_gate(pair, date)
        if not cointegration["passed"]:
            return {
                "passed": False,
                "cointegration": cointegration,
                "mean_reversion": None,
            }

        mean_reversion = self.mean_reversion_gate(cointegration["residual"])
        return {
            "passed": cointegration["passed"] and mean_reversion["passed"],
            "cointegration": cointegration,
            "mean_reversion": mean_reversion,
        }

    def passed_both_gates(self, date):
        passed = []
        results = {}

        for pair in self.active_pairs:
            result = self.regime_filter(pair, date)
            results[self._pair_name(pair)] = result
            if result["passed"]:
                passed.append(pair)

        return passed, results

    def generate_target_signal(self, pair, z_score, regime_passed):
        pair = self._pair_tuple(pair)
        params = self._params(pair)
        state = self.state[pair]
        current_signal = state["raw_signal"]

        if not regime_passed or pd.isna(z_score):
            return 0

        entry_z = params["entry_z"]
        exit_z = params["exit_z"]

        if current_signal == 0:
            if z_score > entry_z:
                return -1
            if z_score < -entry_z:
                return 1
        elif current_signal > 0 and z_score >= -exit_z:
            return 0
        elif current_signal < 0 and z_score <= exit_z:
            return 0

        return current_signal

    def size_by_spread_volatility(self, pair, raw_signal):
        if raw_signal == 0:
            return 0.0

        spread_changes = self.state[pair]["spread_change_history"]
        lookback = int(self._params(pair)["z_lookback"])

        if len(spread_changes) < max(2, lookback):
            return 0.0

        spread_vol = pd.Series(spread_changes[-lookback:]).std()
        if spread_vol == 0 or pd.isna(spread_vol):
            return 0.0

        size = self.target_spread_vol / spread_vol
        size = min(size, self.max_pair_position)
        return raw_signal * size

    def apply_pair_risk_limits(self, target_position):
        return float(np.clip(target_position, -self.max_pair_position, self.max_pair_position))

    def estimate_transaction_cost(self, previous_position, target_position):
        turnover = abs(target_position - previous_position)
        return turnover * self.transaction_cost_bps / 10000

    def process_pair_day(self, pair, date):
        pair = self._pair_tuple(pair)
        state = self.update_kalman_hedge_ratio(pair, date)
        regime = self.regime_filter(pair, date)

        raw_signal = self.generate_target_signal(pair, state["z"], regime["passed"])
        sized_position = self.size_by_spread_volatility(pair, raw_signal)
        target_position = self.apply_pair_risk_limits(sized_position)

        previous_position = state["target_position"]
        transaction_cost = self.estimate_transaction_cost(previous_position, target_position)

        state["raw_signal"] = raw_signal
        state["last_target_position"] = previous_position
        state["target_position"] = target_position

        return {
            "pair": self._pair_name(pair),
            "date": pd.to_datetime(date),
            "alpha": state["alpha"],
            "beta": state["beta"],
            "spread": state["spread"],
            "z": state["z"],
            "regime_passed": regime["passed"],
            "raw_signal": raw_signal,
            "target_position": target_position,
            "previous_position": previous_position,
            "transaction_cost": transaction_cost,
            "adf_pvalue": regime["cointegration"]["adf_pvalue"],
            "half_life": (
                np.nan
                if regime["mean_reversion"] is None
                else regime["mean_reversion"]["half_life"]
            ),
        }

    def apply_portfolio_risk_limits(self, decisions):
        gross_exposure = sum(abs(decision["target_position"]) for decision in decisions)
        if gross_exposure <= self.max_gross_exposure or gross_exposure == 0:
            return decisions

        scale = self.max_gross_exposure / gross_exposure
        for decision in decisions:
            previous_position = decision["previous_position"]
            scaled_position = decision["target_position"] * scale
            decision["target_position"] = scaled_position
            decision["transaction_cost"] = self.estimate_transaction_cost(
                previous_position,
                scaled_position,
            )

            pair = self._pair_tuple(decision["pair"])
            self.state[pair]["target_position"] = scaled_position

        return decisions

    def run_day(self, date):
        decisions = []

        for pair in self.active_pairs:
            decisions.append(self.process_pair_day(pair, date))

        decisions = self.apply_portfolio_risk_limits(decisions)
        return pd.DataFrame(decisions)
