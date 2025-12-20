from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from data import download_close_and_volume

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def _metrics_from_daily_returns(daily_ret: pd.Series) -> dict:
    s = daily_ret.dropna().astype(float)
    n = len(s)
    if n == 0:
        return {"CAGR": np.nan, "AnnReturn": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDrawdown": np.nan}

    equity = (1 + s).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    cagr = (1 + total_return) ** (TRADING_DAYS / n) - 1
    ann_ret = float(s.mean() * TRADING_DAYS)
    vol = float(s.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = float((ann_ret - RISK_FREE_RATE) / vol) if vol > 0 else np.nan
    mdd = _max_drawdown(equity)

    return {"CAGR": float(cagr), "AnnReturn": ann_ret, "Vol": vol, "Sharpe": sharpe, "MaxDrawdown": mdd}


def _get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    """
    Returns rebalance dates as the last available trading date in each period.

    freq examples:
      'M'  = month end
      'Q'  = quarter end
      'W'  = week end
      'A'  = year end
    """
    if len(index) == 0:
        return set()

    periods = index.to_period(freq)
    last_dates = pd.Series(index, index=periods).groupby(level=0).max().values
    return set(pd.to_datetime(last_dates))


def backtest_rebalanced(
    asset_daily_returns: pd.DataFrame,
    target_weights: np.ndarray,
    rebalance_freq: str | None = "M",
    trading_cost_bps: float = 0.0,
    return_details: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Backtest with optional periodic rebalancing to target weights.

    trading_cost_bps:
      - cost applied on rebalance days only
      - cost = turnover * (bps / 10000)
      - turnover = sum(|w_target - w_drift|) on rebalance day

    return_details:
      - False: returns daily return Series
      - True: returns DataFrame with DailyReturn, Turnover, Cost
    """
    r = asset_daily_returns.dropna(how="any").copy()
    idx = r.index
    n = r.shape[1]

    w = np.asarray(target_weights, dtype=float)
    if w.ndim != 1 or len(w) != n:
        raise ValueError("target_weights must be 1D and match number of assets.")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("target_weights must sum to 1.0.")

    # No rebalance = constant weights
    if rebalance_freq is None:
        port = (r * w).sum(axis=1).rename("Portfolio")
        if not return_details:
            return port
        return pd.DataFrame({"DailyReturn": port.values, "Turnover": 0.0, "Cost": 0.0}, index=idx)

    rebalance_dates = _get_rebalance_dates(idx, rebalance_freq)

    portfolio_value = 1.0
    holdings_value = portfolio_value * w

    prev_value = portfolio_value
    port_rets: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []

    cost_rate = float(trading_cost_bps) / 10000.0

    for t in idx:
        # 1) Apply market move
        holdings_value = holdings_value * (1.0 + r.loc[t].to_numpy(dtype=float))
        portfolio_value = float(holdings_value.sum())

        turnover = 0.0
        cost = 0.0

        # 2) Rebalance at period end (and apply costs)
        if t in rebalance_dates:
            w_drift = holdings_value / portfolio_value
            turnover = float(np.sum(np.abs(w - w_drift)))
            cost = float(turnover * cost_rate)

            # Apply cost, then reset holdings to target weights
            portfolio_value = portfolio_value * (1.0 - cost)
            holdings_value = portfolio_value * w

        # 3) Daily return (after any rebalance+cost)
        daily_ret = (portfolio_value / prev_value) - 1.0
        port_rets.append(float(daily_ret))
        turnovers.append(float(turnover))
        costs.append(float(cost))
        prev_value = portfolio_value

    out = pd.Series(port_rets, index=idx, name="Portfolio")

    if not return_details:
        return out

    details = pd.DataFrame(
        {"DailyReturn": out.values, "Turnover": turnovers, "Cost": costs},
        index=idx,
    )
    return details


def run_backtest(
    close_filtered: pd.DataFrame,
    out_dir: Path,
    start_date: str,
    yf_end_exclusive: str,
    benchmark_ticker: str = "^AXJO",
    benchmark_name: str = "Benchmark",
    rebalance_freq: str | None = "M",
    trading_cost_bps: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 3:
    - Build asset daily returns from filtered close
    - Backtest EqualWeight (with optional rebalancing + costs)
    - Download benchmark and compare
    - Save timeseries + summary metrics

    Returns:
      (asset_daily_returns, timeseries_long_df)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if close_filtered.shape[1] < 2:
        print("WARNING: <2 tickers in filtered universe. Optimisation later won’t make sense.")

    # --- Asset daily returns ---
    asset_close = close_filtered.sort_index()
    asset_ret = asset_close.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")

    asset_ret_path = out_dir / "backtest_daily_returns_assets_filtered.csv"
    asset_ret.to_csv(asset_ret_path)
    print(f"Saved: {asset_ret_path}")

    # --- Equal-weight portfolio returns ---
    n = asset_ret.shape[1]
    w_equal = np.ones(n) / n

    eq_details = backtest_rebalanced(
        asset_ret,
        w_equal,
        rebalance_freq=rebalance_freq,
        trading_cost_bps=trading_cost_bps,
        return_details=True,
    )
    port_ret = pd.Series(eq_details["DailyReturn"].values, index=asset_ret.index, name="EqualWeight")

    # --- Benchmark download ---
    print(f"\nDownloading benchmark: {benchmark_ticker} ...")
    bench_close_df, _ = download_close_and_volume(
        tickers=[benchmark_ticker],
        start=start_date,
        end_exclusive=yf_end_exclusive,
    )
    bench_close = bench_close_df.iloc[:, 0].rename(benchmark_name)
    bench_close_path = out_dir / "backtest_benchmark_close.csv"
    bench_close.to_frame().to_csv(bench_close_path)
    print(f"Saved: {bench_close_path}")

    bench_ret = bench_close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    bench_ret.name = benchmark_name

    # --- Align dates ---
    common = asset_ret.index.intersection(bench_ret.index)
    if len(common) == 0:
        raise ValueError("No overlapping dates between portfolio returns and benchmark returns.")

    asset_ret = asset_ret.loc[common]
    port_ret = port_ret.loc[common]
    bench_ret = bench_ret.loc[common]
    eq_details = eq_details.loc[common]

    # --- Equity curves ---
    eq_port = (1 + port_ret).cumprod()
    eq_bench = (1 + bench_ret).cumprod()

    # --- Timeseries (long) ---
    ts = pd.DataFrame({
        "Date": common,
        "Strategy": "EqualWeight",
        "DailyReturn": port_ret.values,
        "Equity": eq_port.values,
        "Turnover": eq_details["Turnover"].values,
        "Cost": eq_details["Cost"].values,
        "IsBenchmark": False,
    })

    ts_b = pd.DataFrame({
        "Date": common,
        "Strategy": benchmark_name,
        "DailyReturn": bench_ret.values,
        "Equity": eq_bench.values,
        "Turnover": 0.0,
        "Cost": 0.0,
        "IsBenchmark": True,
    })

    ts_long = pd.concat([ts, ts_b], ignore_index=True)

    ts_path = out_dir / "backtest_timeseries_equal_vs_benchmark.csv"
    ts_long.to_csv(ts_path, index=False)
    print(f"Saved: {ts_path}")

    # --- Summary metrics ---
    m_port = _metrics_from_daily_returns(port_ret)
    m_bench = _metrics_from_daily_returns(bench_ret)

    metrics_df = pd.DataFrame([
        {"Strategy": "EqualWeight", "IsBenchmark": False, **m_port},
        {"Strategy": benchmark_name, "IsBenchmark": True, **m_bench},
    ])
    metrics_path = out_dir / "backtest_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    print("\nBacktest complete.")
    print(f"Rebalance frequency used: {rebalance_freq}")
    print(f"Trading cost (bps) used: {trading_cost_bps}")
    print(f"Date range used: {common.min().date()} to {common.max().date()}")
    print(f"Assets in filtered universe: {n}")

    return asset_ret, ts_long


def run_strategy_backtests(
    asset_daily_returns: pd.DataFrame,
    weights: dict[str, np.ndarray],
    out_dir: Path,
    start_date: str,
    yf_end_exclusive: str,
    benchmark_ticker: str = "^AXJO",
    benchmark_name: str = "Benchmark",
    rebalance_freq: str | None = "M",
    trading_cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 5:
    Backtest multiple strategies (weights dict) vs a benchmark.
    Saves:
      - strategies_timeseries.csv (long format)
      - strategies_metrics_summary.csv
    Returns:
      (timeseries_long, metrics_df)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Benchmark download ---
    print(f"\nDownloading benchmark: {benchmark_ticker} ...")
    bench_close_df, _ = download_close_and_volume(
        tickers=[benchmark_ticker],
        start=start_date,
        end_exclusive=yf_end_exclusive,
    )
    bench_close = bench_close_df.iloc[:, 0].rename(benchmark_name)
    bench_ret = bench_close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    bench_ret.name = benchmark_name

    # --- Align dates across strategies + benchmark ---
    common = asset_daily_returns.index.intersection(bench_ret.index)
    if len(common) == 0:
        raise ValueError("No overlapping dates between strategy returns and benchmark returns.")

    asset_ret = asset_daily_returns.loc[common]
    bench_ret = bench_ret.loc[common]

    # --- Build daily returns per strategy (with costs) ---
    strat_details: dict[str, pd.DataFrame] = {}
    for name, w in weights.items():
        details = backtest_rebalanced(
            asset_ret,
            w,
            rebalance_freq=rebalance_freq,
            trading_cost_bps=trading_cost_bps,
            return_details=True,
        )
        strat_details[name] = details

    # --- Timeseries (long) with equity ---
    ts_parts = []

    for name, details in strat_details.items():
        dr = pd.Series(details["DailyReturn"].values, index=common, name=name)
        equity = (1 + dr).cumprod()

        tmp = pd.DataFrame({
            "Date": common,
            "Strategy": name,
            "DailyReturn": dr.values,
            "Equity": equity.values,
            "Turnover": details["Turnover"].values,
            "Cost": details["Cost"].values,
            "IsBenchmark": False,
        })
        ts_parts.append(tmp)

    # Add benchmark
    eq_bench = (1 + bench_ret).cumprod()
    ts_parts.append(pd.DataFrame({
        "Date": common,
        "Strategy": benchmark_name,
        "DailyReturn": bench_ret.values,
        "Equity": eq_bench.values,
        "Turnover": 0.0,
        "Cost": 0.0,
        "IsBenchmark": True,
    }))

    ts_long = pd.concat(ts_parts, ignore_index=True)

    ts_path = out_dir / "strategies_timeseries.csv"
    ts_long.to_csv(ts_path, index=False)
    print(f"Saved: {ts_path}")

    # --- Metrics summary ---
    metric_rows = []
    for name, details in strat_details.items():
        s = pd.Series(details["DailyReturn"].values, index=common, name=name)
        m = _metrics_from_daily_returns(s)
        metric_rows.append({"Strategy": name, "IsBenchmark": False, **m})

    m_bench = _metrics_from_daily_returns(bench_ret)
    metric_rows.append({"Strategy": benchmark_name, "IsBenchmark": True, **m_bench})

    metrics_df = pd.DataFrame(metric_rows).sort_values("Sharpe", ascending=False)

    metrics_path = out_dir / "strategies_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    print("\nStrategy backtests complete.")
    print(f"Rebalance frequency used: {rebalance_freq}")
    print(f"Trading cost (bps) used: {trading_cost_bps}")
    print(f"Date range used: {common.min().date()} to {common.max().date()}")
    print("Strategies:", ", ".join(list(weights.keys()) + [benchmark_name]))

    return ts_long, metrics_df
