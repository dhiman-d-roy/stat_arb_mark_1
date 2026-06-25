import numpy as np
import pandas as pd


def kalman_filter_beta(y, x, q_alpha=1e-5, q_beta=1e-5, r=1e-3):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    n = len(y)

    alpha_series = np.full(n, np.nan)
    beta_series = np.full(n, np.nan)
    spread_series = np.full(n, np.nan)

    theta = np.array([0.0, 1.0])
    p_cov = np.eye(2)
    q_cov = np.array([
        [q_alpha, 0.0],
        [0.0, q_beta],
    ])

    for t in range(n):
        h_obs = np.array([1.0, x[t]])

        theta_pred = theta
        p_pred = p_cov + q_cov

        y_pred = h_obs @ theta_pred
        innovation = y[t] - y_pred
        innovation_var = h_obs @ p_pred @ h_obs.T + r

        kalman_gain = p_pred @ h_obs.T / innovation_var
        theta = theta_pred + kalman_gain * innovation
        p_cov = p_pred - np.outer(kalman_gain, h_obs) @ p_pred

        alpha_series[t] = theta[0]
        beta_series[t] = theta[1]
        spread_series[t] = y[t] - theta[0] - theta[1] * x[t]

    return pd.DataFrame({
        "alpha_kalman": alpha_series,
        "beta_kalman": beta_series,
        "spread": spread_series,
    })


def generate_positions(df, entry_z, exit_z):
    result = df.copy()
    positions = []
    current_pos = 0

    for z_score in result["z"]:
        if pd.isna(z_score):
            positions.append(current_pos)
            continue

        if current_pos == 0:
            if z_score > entry_z:
                current_pos = -1
            elif z_score < -entry_z:
                current_pos = 1
        elif current_pos > 0 and z_score >= -exit_z:
            current_pos = 0
        elif current_pos < 0 and z_score <= exit_z:
            current_pos = 0

        positions.append(current_pos)

    result["position"] = positions
    return result


class Tuner:
    KALMAN_PARAM_GRID = [
        {"q_alpha": 1e-6, "q_beta": 1e-6, "r": 1e-3},
        {"q_alpha": 1e-5, "q_beta": 1e-6, "r": 1e-3},
        {"q_alpha": 1e-6, "q_beta": 1e-5, "r": 1e-3},
        {"q_alpha": 1e-5, "q_beta": 1e-5, "r": 1e-3},
        {"q_alpha": 1e-4, "q_beta": 1e-5, "r": 1e-3},
        {"q_alpha": 1e-5, "q_beta": 1e-4, "r": 1e-3},
        {"q_alpha": 1e-4, "q_beta": 1e-4, "r": 1e-3},
        {"q_alpha": 1e-5, "q_beta": 1e-5, "r": 1e-4},
        {"q_alpha": 1e-5, "q_beta": 1e-5, "r": 1e-2},
    ]

    LOOKBACK_GRID = np.array(list(range(2, 63)))
    ENTRY_Z_GRID = np.array(list(range(5, 21))) / 10
    EXIT_Z_GRID = np.array(list(range(1, 16))) / 10

    Z_LOOKBACK = 63
    ENTRY_Z = 1.5
    EXIT_Z = 0.5
    MIN_ENTRIES = 20
    MIN_SHARPE = 1

    def __init__(self, df, pair, start_date, end_date):
        self.df = df.copy()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.pair = pair
        self.y_ticker = pair[0]
        self.x_ticker = pair[1]

        self.q_alpha = None
        self.q_beta = None
        self.r = None
        self.z_lookback = None
        self.entry_z = None
        self.exit_z = None

        self.kalman_tuning_results = None
        self.lookback_tuning_results = None
        self.entry_exit_tuning_results = None
        self.tuned_params = None
        self.final_backtest = None

    def prepare_kalman_set(self, q_alpha, q_beta, r, z_lookback=None):
        if z_lookback is None:
            z_lookback = self.Z_LOOKBACK

        result = self.df.copy().reset_index(drop=True)
        mask = (result["Date"] >= self.start_date) & (result["Date"] <= self.end_date)
        result = result.loc[mask].copy().reset_index(drop=True)

        if result.empty:
            raise ValueError("No rows found between start_date and end_date.")

        kalman_result = kalman_filter_beta(
            y=result[f"log_{self.y_ticker}"],
            x=result[f"log_{self.x_ticker}"],
            q_alpha=q_alpha,
            q_beta=q_beta,
            r=r,
        )

        result = pd.concat([result, kalman_result], axis=1)
        result["spread_mean"] = result["spread"].rolling(z_lookback).mean().shift(1)
        result["spread_std"] = result["spread"].rolling(z_lookback).std().shift(1)
        result["z"] = (result["spread"] - result["spread_mean"]) / result["spread_std"]
        return result

    def _select_best(self, results, label):
        if results.empty:
            raise ValueError(f"No {label} results were generated.")

        ranked = results.copy()
        ranked["passes_filter"] = (
            (ranked["entries"] >= self.MIN_ENTRIES)
            & (ranked["sharpe"] > self.MIN_SHARPE)
        )

        ranked = (
            ranked
            .sort_values(["sharpe", "total_pnl"], ascending=[False, False], na_position="last")
            .reset_index(drop=True)
        )

        candidates = ranked[ranked["passes_filter"]].copy()
        if candidates.empty:
            candidates = ranked.copy()

        if candidates.empty or pd.isna(candidates.iloc[0]["sharpe"]):
            raise ValueError(f"No usable {label} results were generated.")

        return ranked, candidates.iloc[0]

    def run_spread_backtest(
        self,
        q_alpha=None,
        q_beta=None,
        r=None,
        entry_z=None,
        exit_z=None,
        z_lookback=None,
    ):
        if q_alpha is None:
            q_alpha = self.q_alpha if self.q_alpha is not None else 1e-5
        if q_beta is None:
            q_beta = self.q_beta if self.q_beta is not None else 1e-5
        if r is None:
            r = self.r if self.r is not None else 1e-3
        if z_lookback is None:
            z_lookback = self.z_lookback if self.z_lookback is not None else self.Z_LOOKBACK
        if entry_z is None:
            entry_z = self.entry_z if self.entry_z is not None else self.ENTRY_Z
        if exit_z is None:
            exit_z = self.exit_z if self.exit_z is not None else self.EXIT_Z

        result = self.prepare_kalman_set(
            q_alpha=q_alpha,
            q_beta=q_beta,
            r=r,
            z_lookback=z_lookback,
        )

        result = generate_positions(result, entry_z=entry_z, exit_z=exit_z)

        result["spread_change"] = result["spread"].diff()
        result["pnl"] = result["position"].shift(1).fillna(0) * result["spread_change"]
        result["cum_pnl"] = result["pnl"].cumsum()
        result["drawdown"] = result["cum_pnl"] - result["cum_pnl"].cummax()

        daily_vol = result["pnl"].std()
        if daily_vol == 0 or pd.isna(daily_vol):
            sharpe = np.nan
        else:
            sharpe = np.sqrt(252) * result["pnl"].mean() / daily_vol

        entries = (
            (result["position"] != 0)
            & (result["position"].shift(1).fillna(0) == 0)
        ).sum()

        return {
            "q_alpha": q_alpha,
            "q_beta": q_beta,
            "r": r,
            "z_lookback": z_lookback,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "total_pnl": result["pnl"].sum(),
            "annual_pnl": result["pnl"].mean() * 252,
            "daily_vol": daily_vol,
            "sharpe": sharpe,
            "max_drawdown": result["drawdown"].min(),
            "entries": int(entries),
            "exposure": result["position"].ne(0).mean(),
            "backtest_df": result,
        }

    def tune_kalman_parameters(self, entry_z=None, exit_z=None, z_lookback=None):
        tuning_results = []

        for kalman_params in self.KALMAN_PARAM_GRID:
            stats = self.run_spread_backtest(
                q_alpha=kalman_params["q_alpha"],
                q_beta=kalman_params["q_beta"],
                r=kalman_params["r"],
                entry_z=entry_z,
                exit_z=exit_z,
                z_lookback=z_lookback,
            )
            tuning_results.append({k: v for k, v in stats.items() if k != "backtest_df"})

        tuning_results = pd.DataFrame(tuning_results)
        ranked, best = self._select_best(tuning_results, "Kalman parameter")

        self.q_alpha = best["q_alpha"]
        self.q_beta = best["q_beta"]
        self.r = best["r"]
        self.kalman_tuning_results = ranked

        return {
            "q_alpha": self.q_alpha,
            "q_beta": self.q_beta,
            "r": self.r,
        }

    def run_spread_backtest_lookback_z(
        self,
        q_alpha,
        q_beta,
        r,
        z_lookback,
        entry_z=None,
        exit_z=None,
    ):
        return self.run_spread_backtest(
            q_alpha=q_alpha,
            q_beta=q_beta,
            r=r,
            z_lookback=z_lookback,
            entry_z=entry_z,
            exit_z=exit_z,
        )

    def tune_lookback(self, entry_z=None, exit_z=None):
        if self.q_alpha is None or self.q_beta is None or self.r is None:
            self.tune_kalman_parameters(entry_z=entry_z, exit_z=exit_z)

        lookback_results = []

        for lookback in self.LOOKBACK_GRID:
            stats = self.run_spread_backtest(
                q_alpha=self.q_alpha,
                q_beta=self.q_beta,
                r=self.r,
                z_lookback=lookback,
                entry_z=entry_z,
                exit_z=exit_z,
            )
            lookback_results.append({k: v for k, v in stats.items() if k != "backtest_df"})

        lookback_results = pd.DataFrame(lookback_results)
        ranked, best = self._select_best(lookback_results, "lookback")

        self.z_lookback = int(best["z_lookback"])
        self.lookback_tuning_results = ranked

        return {"z_lookback": self.z_lookback}

    def run_spread_backtest_entry_exit_z(
        self,
        q_alpha,
        q_beta,
        r,
        z_lookback,
        entry_z,
        exit_z,
    ):
        return self.run_spread_backtest(
            q_alpha=q_alpha,
            q_beta=q_beta,
            r=r,
            z_lookback=z_lookback,
            entry_z=entry_z,
            exit_z=exit_z,
        )

    def tune_entry_and_exit_z(self, z_lookback=None):
        if self.q_alpha is None or self.q_beta is None or self.r is None:
            self.tune_kalman_parameters()

        if z_lookback is None:
            if self.z_lookback is None:
                self.tune_lookback()
            z_lookback = self.z_lookback

        results = []

        for entry_z in self.ENTRY_Z_GRID:
            for exit_z in self.EXIT_Z_GRID:
                if exit_z >= entry_z:
                    continue

                stats = self.run_spread_backtest(
                    q_alpha=self.q_alpha,
                    q_beta=self.q_beta,
                    r=self.r,
                    z_lookback=z_lookback,
                    entry_z=entry_z,
                    exit_z=exit_z,
                )
                results.append({k: v for k, v in stats.items() if k != "backtest_df"})

        results = pd.DataFrame(results)
        ranked, best = self._select_best(results, "entry/exit z-score")

        self.entry_z = best["entry_z"]
        self.exit_z = best["exit_z"]
        self.entry_exit_tuning_results = ranked

        return {
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
        }

    def tune(self):
        kalman_params = self.tune_kalman_parameters()
        lookback_params = self.tune_lookback()
        z_params = self.tune_entry_and_exit_z(z_lookback=lookback_params["z_lookback"])

        final_stats = self.run_spread_backtest(
            q_alpha=kalman_params["q_alpha"],
            q_beta=kalman_params["q_beta"],
            r=kalman_params["r"],
            z_lookback=lookback_params["z_lookback"],
            entry_z=z_params["entry_z"],
            exit_z=z_params["exit_z"],
        )

        self.final_backtest = final_stats["backtest_df"]
        self.tuned_params = {
            "pair": f"{self.y_ticker}/{self.x_ticker}",
            "start_date": self.start_date,
            "end_date": self.end_date,
            "q_alpha": kalman_params["q_alpha"],
            "q_beta": kalman_params["q_beta"],
            "r": kalman_params["r"],
            "z_lookback": lookback_params["z_lookback"],
            "entry_z": z_params["entry_z"],
            "exit_z": z_params["exit_z"],
            "sharpe": final_stats["sharpe"],
            "total_pnl": final_stats["total_pnl"],
            "annual_pnl": final_stats["annual_pnl"],
            "daily_vol": final_stats["daily_vol"],
            "max_drawdown": final_stats["max_drawdown"],
            "entries": final_stats["entries"],
            "exposure": final_stats["exposure"],
        }

        return self.tuned_params

    @classmethod
    def tune_universe(
        cls,
        pair_data,
        pairs=None,
        end_date=None,
        trailing_window_days=252,
        min_sharpe=None,
        min_entries=None,
        max_pairs=None,
    ):
        if pairs is None:
            pairs = list(pair_data.keys())

        if end_date is None:
            end_date = min(pair_data[pair]["Date"].max() for pair in pairs)

        end_date = pd.to_datetime(end_date)
        start_date = end_date - pd.DateOffset(days=trailing_window_days)
        min_sharpe = cls.MIN_SHARPE if min_sharpe is None else min_sharpe
        min_entries = cls.MIN_ENTRIES if min_entries is None else min_entries

        tuned_rows = []
        tuners = {}
        failures = {}

        for pair in pairs:
            tuner = cls(
                df=pair_data[pair],
                pair=pair,
                start_date=start_date,
                end_date=end_date,
            )

            try:
                tuned_params = tuner.tune()
            except ValueError as exc:
                failures[pair] = str(exc)
                continue

            tuned_rows.append(tuned_params)
            tuners[pair] = tuner

        if not tuned_rows:
            return {
                "active_pairs": [],
                "tuned_params": pd.DataFrame(),
                "tuners": tuners,
                "failures": failures,
            }

        tuned_params = pd.DataFrame(tuned_rows)
        tuned_params["passes_filter"] = (
            (tuned_params["entries"] >= min_entries)
            & (tuned_params["sharpe"] > min_sharpe)
        )

        active_params = (
            tuned_params[tuned_params["passes_filter"]]
            .sort_values(["sharpe", "total_pnl"], ascending=[False, False], na_position="last")
            .reset_index(drop=True)
        )

        if max_pairs is not None:
            active_params = active_params.head(max_pairs).copy()

        active_pairs = [tuple(pair_name.split("/")) for pair_name in active_params["pair"]]

        return {
            "active_pairs": active_pairs,
            "tuned_params": active_params,
            "all_tuned_params": tuned_params,
            "tuners": tuners,
            "failures": failures,
        }


Tuning = Tuner
