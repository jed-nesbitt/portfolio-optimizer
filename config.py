from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta, datetime
from pathlib import Path


@dataclass(frozen=True)
class Config:
    start_date: str
    asof_date_inclusive: str
    yf_end_exclusive: str
    tickers_csv: Path
    out_dir: Path  # <- now this is outputs/<run_id>/
    rebalance_freq: str | None
    trading_cost_bps: float  # e.g. 10 = 0.10%


def build_config() -> Config:
    asof = date.today() - timedelta(days=1)
    yf_end = date.today()  # yfinance end is exclusive

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / run_id
    

    return Config(
        start_date="2018-01-01",
        asof_date_inclusive=asof.isoformat(),
        yf_end_exclusive=yf_end.isoformat(),
        tickers_csv=Path("input") / "tickers.csv",
        out_dir=out_dir,
        rebalance_freq="M",
        trading_cost_bps=10.0,   # set 0.0 for no costs

    )
