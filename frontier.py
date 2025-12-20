from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS_DEFAULT = 252


def _expected_mu_cov(asset_daily_returns: pd.DataFrame, trading_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert daily returns to annualised mean + covariance.
    """
    mu = (asset_daily_returns.mean() * trading_days).to_numpy(dtype=float)
    cov = (asset_daily_returns.cov() * trading_days).to_numpy(dtype=float)
    return mu, cov


def _port_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> Tuple[float, float, float]:
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu)
    vol = float(np.sqrt(w.T @ cov @ w))
    sharpe = float((ret - rf) / vol) if vol > 0 else np.nan
    return ret, vol, sharpe


def _random_weights_with_cap(rng: np.random.Generator, n: int, max_weight: float) -> np.ndarray | None:
    """
    Generates random long-only weights that sum to 1, with each weight <= max_weight.
    Returns None if generated vector violates cap.
    """
    w = rng.random(n)
    w = w / w.sum()

    if max_weight < 1.0 and np.any(w > max_weight):
        return None
    return w


def generate_frontier_cloud(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    num_portfolios: int,
    seed: int,
    max_weight: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generates a random portfolio cloud to visualise the frontier.
    Returns:
      frontier_df (ExpectedReturn, Volatility, Sharpe, PortfolioID)
      weights_array (num_kept x n_assets)
    """
    rng = np.random.default_rng(seed)
    n = len(mu)

    rets, vols, sharpes, weights_list = [], [], [], []
    attempts = 0
    max_attempts = max(50_000, num_portfolios * 50)

    # Sanity: cap must allow feasible solution
    if max_weight < 1.0 / n:
        raise ValueError(f"max_weight={max_weight} is too small for n={n}. Must be >= {1.0/n:.4f}")

    while len(weights_list) < num_portfolios and attempts < max_attempts:
        attempts += 1
        w = _random_weights_with_cap(rng, n, max_weight)
        if w is None:
            continue

        r, v, s = _port_stats(w, mu, cov, rf)
        rets.append(r)
        vols.append(v)
        sharpes.append(s)
        weights_list.append(w)

    weights_array = np.array(weights_list, dtype=float)

    frontier_df = pd.DataFrame({
        "PortfolioID": np.arange(len(weights_array)),
        "ExpectedReturn": rets,
        "Volatility": vols,
        "Sharpe": sharpes,
    })

    return frontier_df, weights_array


def pick_optimals_from_cloud(frontier_df: pd.DataFrame, weights_array: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Choose MinVol/MaxSharpe from random cloud.
    """
    vol = frontier_df["Volatility"].to_numpy()
    sh = frontier_df["Sharpe"].to_numpy()

    min_vol_idx = int(np.nanargmin(vol))
    max_sh_idx = int(np.nanargmax(sh))

    return {
        "MinVol": weights_array[min_vol_idx],
        "MaxSharpe": weights_array[max_sh_idx],
    }
def pick_min_vol_for_target_return(
    frontier_df: pd.DataFrame,
    weights_array: np.ndarray,
    target_return: float,
    tolerance: float = 0.002,  # 0.2% band
) -> np.ndarray:
    """
    Pick the minimum-volatility portfolio that matches a target expected return.

    - First tries portfolios with ExpectedReturn within +/- tolerance of target_return.
    - If none exist, falls back to portfolios closest to the target (by abs difference),
      then chooses the lowest volatility among the closest group.
    """
    er = frontier_df["ExpectedReturn"].to_numpy(dtype=float)
    vol = frontier_df["Volatility"].to_numpy(dtype=float)

    # 1) Try within band
    in_band = np.where(np.abs(er - target_return) <= tolerance)[0]
    if len(in_band) > 0:
        best_idx = int(in_band[np.argmin(vol[in_band])])
        return weights_array[best_idx]

    # 2) Fallback: take closest-by-return portfolios, then min vol
    abs_diff = np.abs(er - target_return)
    min_diff = np.nanmin(abs_diff)
    closest = np.where(abs_diff == min_diff)[0]

    best_idx = int(closest[np.argmin(vol[closest])])
    return weights_array[best_idx]


def save_frontier_plot(frontier_df: pd.DataFrame, out_path: Path, special_points: Dict[str, Tuple[float, float]]) -> None:
    """
    Saves a simple frontier plot. (Uses matplotlib; no fancy styling.)
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.scatter(frontier_df["Volatility"], frontier_df["ExpectedReturn"], c=frontier_df["Sharpe"], s=10, alpha=0.6)
    plt.colorbar(label="Sharpe Ratio")

    for name, (v, r) in special_points.items():
        plt.scatter(v, r, s=250, marker="*", label=name)

    plt.xlabel("Volatility (ann.)")
    plt.ylabel("Expected Return (ann.)")
    plt.title("Efficient Frontier (Random Cloud + Selected Portfolios)")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_frontier(
    asset_daily_returns: pd.DataFrame,
    out_dir: Path,
    rf: float = 0.04,
    trading_days: int = TRADING_DAYS_DEFAULT,
    num_portfolios: int = 15000,
    seed: int = 42,
    max_weight: float = 0.2,
    target_return: float | None = None,
    target_tolerance: float = 0.002,
) -> Dict[str, np.ndarray]:

    """
    Frontier runner (NO SciPy):
    - compute mu/cov
    - generate random cloud
    - select MaxSharpe and MinVol from the cloud
    - export frontier + weights + summary + plot
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if asset_daily_returns.shape[1] < 2:
        raise ValueError("Need at least 2 assets for a frontier.")

    tickers = list(asset_daily_returns.columns)

    mu, cov = _expected_mu_cov(asset_daily_returns, trading_days)

    # 1) Cloud
    frontier_df, weights_array = generate_frontier_cloud(
        mu=mu,
        cov=cov,
        rf=rf,
        num_portfolios=num_portfolios,
        seed=seed,
        max_weight=max_weight,
    )
    frontier_path = out_dir / "frontier_points.csv"
    frontier_df.to_csv(frontier_path, index=False)
    print(f"Saved: {frontier_path}")

    # 2) Selected portfolios from the cloud
    optimals = pick_optimals_from_cloud(frontier_df, weights_array)
    print("Optimisation: selected from random cloud (SciPy disabled).")

    # Target return min-vol (optional)
    if target_return is not None:
        w_target = pick_min_vol_for_target_return(
            frontier_df, weights_array, target_return=target_return, tolerance=target_tolerance
        )
        optimals[f"TargetReturn_{target_return:.2%}"] = w_target
        print(f"Added strategy: TargetReturn_{target_return:.2%} (min vol near target).")
    # Add equal weight
    n = len(tickers)
    optimals["EqualWeight"] = np.ones(n) / n

    # 3) Save weights
    rows = {"Ticker": tickers}
    for name, w in optimals.items():
        rows[f"{name}_Weight"] = np.asarray(w, dtype=float)

    weights_df = pd.DataFrame(rows)
    weights_out = out_dir / "frontier_optimal_weights.csv"
    weights_df.to_csv(weights_out, index=False)
    print(f"Saved: {weights_out}")

    # 4) Summary (expected stats)
    summary_rows = []
    for name, w in optimals.items():
        r, v, s = _port_stats(w, mu, cov, rf)
        summary_rows.append({"Strategy": name, "ExpectedReturn": r, "Volatility": v, "Sharpe": s})

    summary_df = pd.DataFrame(summary_rows).sort_values("Sharpe", ascending=False)
    summary_out = out_dir / "frontier_expected_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved: {summary_out}")

    # 5) Plot
    special = {}
    for name, w in optimals.items():
        r, v, _ = _port_stats(w, mu, cov, rf)
        special[name] = (v, r)

    plot_out = out_dir / "efficient_frontier.png"
    save_frontier_plot(frontier_df, plot_out, special_points=special)
    print(f"Saved: {plot_out}")

    return optimals
