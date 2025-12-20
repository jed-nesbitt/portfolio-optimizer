from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScreenRules:
    adv_window: int = 60            # trading days for liquidity
    min_price: float = 0.10         # drop penny stocks below this
    min_adv_dollars: float = 500_000  # avg daily $ volume threshold
    max_zero_ret_frac: float = 0.20 # drop if >20% of days have ~0 return (stale)
    zero_ret_eps: float = 1e-12     # tolerance for "zero return"


def build_screening_report(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    rules: ScreenRules,
) -> pd.DataFrame:
    """
    Returns a per-ticker report with columns:
    LatestPrice, ADV_$, ZeroRetFrac, NumObs, FirstDate, LastDate, Keep
    """
    # Align
    close = close.sort_index()
    volume = volume.reindex(close.index)

    # Basic stats
    latest_price = close.iloc[-1]
    num_obs = close.notna().sum(axis=0)
    first_date = close.apply(lambda s: s.first_valid_index())
    last_date = close.apply(lambda s: s.last_valid_index())

    # Liquidity: ADV in dollars
    dollar_vol = close * volume
    adv = dollar_vol.tail(rules.adv_window).mean(axis=0, skipna=True)

    # Staleness: fraction of near-zero daily returns
    daily_ret = close.pct_change()
    zero_ret_frac = (daily_ret.abs() < rules.zero_ret_eps).mean(axis=0, skipna=True)

    report = pd.DataFrame({
        "LatestPrice": latest_price,
        "ADV_$": adv,
        "ZeroRetFrac": zero_ret_frac,
        "NumObs": num_obs,
        "FirstDate": first_date.astype("datetime64[ns]"),
        "LastDate": last_date.astype("datetime64[ns]"),
    })

    # Keep rule
    report["Keep"] = (
        (report["LatestPrice"] >= rules.min_price) &
        (report["ADV_$"] >= rules.min_adv_dollars) &
        (report["ZeroRetFrac"] <= rules.max_zero_ret_frac)
    )

    # Helpful: reason flags
    report["Fail_Price"] = report["LatestPrice"] < rules.min_price
    report["Fail_ADV"] = report["ADV_$"] < rules.min_adv_dollars
    report["Fail_Stale"] = report["ZeroRetFrac"] > rules.max_zero_ret_frac

    # Sort: worst offenders first
    report = report.sort_values(["Keep", "ADV_$", "LatestPrice"], ascending=[True, True, True])

    return report


def apply_screen(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    report: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_cols = report.index[report["Keep"]].tolist()
    close_f = close[keep_cols].copy()
    vol_f = volume[keep_cols].copy()
    return close_f, vol_f
