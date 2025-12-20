from __future__ import annotations

from config import build_config
from data import load_tickers, download_close_and_volume
from screening import ScreenRules, build_screening_report, apply_screen
from backtest import run_backtest, run_strategy_backtests
from frontier import run_frontier


def main() -> None:
    cfg = build_config()

    # Create per-step output folders
    raw_dir = cfg.out_dir / "01_raw"
    screen_dir = cfg.out_dir / "02_screening"
    bt_dir = cfg.out_dir / "03_backtest"
    frontier_dir = cfg.out_dir / "04_frontier"
    strategies_dir = cfg.out_dir / "05_strategies"

    for d in [raw_dir, screen_dir, bt_dir, frontier_dir, strategies_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nAll outputs will be saved under: {cfg.out_dir.resolve()}\n")

    print("=== STEP 1: Download prices ===")
    print(f"Tickers file: {cfg.tickers_csv}")
    print(f"Start date: {cfg.start_date}")
    print(f"Target last included date (yesterday): {cfg.asof_date_inclusive}")
    print(f"yfinance end (exclusive): {cfg.yf_end_exclusive}")

    tickers = load_tickers(cfg.tickers_csv)
    if len(tickers) == 0:
        raise ValueError("No tickers found in input/tickers.csv")

    print(f"\nDownloading {len(tickers)} tickers...\n")
    close, volume = download_close_and_volume(
        tickers=tickers,
        start=cfg.start_date,
        end_exclusive=cfg.yf_end_exclusive,
    )

    # Save raw
    raw_close_path = raw_dir / "raw_close.csv"
    raw_vol_path = raw_dir / "raw_volume.csv"
    close.to_csv(raw_close_path)
    volume.to_csv(raw_vol_path)

    last_date = close.index.max().date().isoformat()
    print(f"\nLast available trading date in data: {last_date}")
    print(f"Raw saved:\n- {raw_close_path}\n- {raw_vol_path}")

    # === STEP 2: Screening ===
    print("\n=== STEP 2: Investability screening ===")

    rules = ScreenRules(
        adv_window=60,
        min_price=0.01,
        min_adv_dollars=50_000,
        max_zero_ret_frac=0.20,
    )

    report = build_screening_report(close, volume, rules)
    report_path = screen_dir / "universe_screening_report.csv"
    report.to_csv(report_path)

    kept = int(report["Keep"].sum())
    dropped = int((~report["Keep"]).sum())
    print(f"Kept: {kept} | Dropped: {dropped}")
    print(f"Saved screening report: {report_path}")

    close_f, vol_f = apply_screen(close, volume, report)

    close_f_path = screen_dir / "close_filtered.csv"
    vol_f_path = screen_dir / "volume_filtered.csv"
    close_f.to_csv(close_f_path)
    vol_f.to_csv(vol_f_path)

    print(f"Saved filtered data:\n- {close_f_path}\n- {vol_f_path}")

    if kept < 2:
        print("\nWARNING: <2 tickers left after screening.")
        print("To keep more, relax rules (lower min_adv_dollars / min_price, or raise max_zero_ret_frac).")
        return

    # === STEP 3: Backtest ===
    print("\n=== STEP 3: Backtest Equal Weight vs Benchmark ===")
    asset_ret, _ = run_backtest(
        close_filtered=close_f,
        out_dir=bt_dir,
        start_date=cfg.start_date,
        yf_end_exclusive=cfg.yf_end_exclusive,
        benchmark_ticker="^AXJO",
        benchmark_name="Benchmark",
        rebalance_freq=cfg.rebalance_freq,
        trading_cost_bps=cfg.trading_cost_bps,
    )

    # Target-return strategy (optional)
    target_str = input(
        "\nOptional: enter target annual expected return (e.g. 0.10 for 10%). Press Enter to skip: "
    ).strip()
    target_return = float(target_str) if target_str else None

    # === STEP 4: Efficient Frontier ===
    print("\n=== STEP 4: Efficient Frontier ===")
    optimals = run_frontier(
        asset_daily_returns=asset_ret,
        out_dir=frontier_dir,
        rf=0.04,
        trading_days=252,
        num_portfolios=15000,
        seed=42,
        max_weight=0.2,
        target_return=target_return,
        target_tolerance=0.002,
    )

    # === STEP 5: Backtest frontier portfolios ===
    print("\n=== STEP 5: Backtest frontier portfolios vs benchmark ===")
    run_strategy_backtests(
        asset_daily_returns=asset_ret,
        weights=optimals,
        out_dir=strategies_dir,
        start_date=cfg.start_date,
        yf_end_exclusive=cfg.yf_end_exclusive,
        benchmark_ticker="^AXJO",
        benchmark_name="Benchmark",
        rebalance_freq=cfg.rebalance_freq,
        trading_cost_bps=cfg.trading_cost_bps,
    )


if __name__ == "__main__":
    main()
