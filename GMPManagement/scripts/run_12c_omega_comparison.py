import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils.config import project_root, get_config
from src.utils.logger import setup_logger
from src.data.loader import load_processed
from src.gmp.analytics import get_class_returns_and_weights
from src.black_litterman.bl_backtest import bl_walk_forward
from src.analytics.performance import summary
from src.analytics.returns import log_to_simple


def _normalize_index(s):
    out = s.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.astype("datetime64[ns]")
    return out


def build_signals(returns_monthly, include_carry=True, include_value=True):
    signals = {}
    try:
        from src.signals.credit import credit_to_view
        c = credit_to_view(returns_monthly)
        if not c.empty:
            signals["credit"] = _normalize_index(c)
    except Exception as e:
        print(f"credit: {e}")
    try:
        from src.signals.vix import fetch_vix_term_structure, vix_to_view
        v = fetch_vix_term_structure(start="2011-01-01")
        if not v.empty:
            vv = vix_to_view(v)
            if not vv.empty:
                signals["vix"] = _normalize_index(vv)
    except Exception as e:
        print(f"vix: {e}")
    try:
        from src.signals.inflation import term_premium_view
        tp = term_premium_view(start="2000-01-01")
        if not tp.empty:
            signals["term_premium"] = _normalize_index(tp)
    except Exception as e:
        print(f"term_premium: {e}")
    if include_carry:
        try:
            from src.signals.carry import carry_to_view
            cr = carry_to_view(start="2000-01-01")
            if not cr.empty:
                signals["carry"] = _normalize_index(cr)
        except Exception as e:
            print(f"carry: {e}")
    try:
        from src.signals.trend import trend_to_view
        asset = "ACWI" if "ACWI" in returns_monthly.columns else None
        if asset:
            t = trend_to_view(returns_monthly, asset=asset, lookback=12)
            if not t.empty:
                signals["trend"] = _normalize_index(t)
    except Exception as e:
        print(f"trend: {e}")
    if include_value:
        try:
            from src.signals.value import value_to_view
            v = value_to_view(lookback=120)
            if not v.empty:
                signals["value"] = _normalize_index(v)
        except Exception as e:
            print(f"value: {e}")
    return signals


def main():
    cfg = get_config()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    with open(ROOT / "config" / "assets.yaml", "r") as f:
        assets_cfg = yaml.safe_load(f)
    ticker_to_gmp = assets_cfg["ticker_to_gmp"]

    returns = load_processed("returns_monthly")
    weights_gmp = pd.read_parquet(
        project_root() / "data/processed/gmp_weights_history.parquet"
    )

    rf_ticker = cfg["data"]["risk_free_ticker"]
    rf_simple = _normalize_index(log_to_simple(returns[rf_ticker])) \
        if rf_ticker in returns.columns else None
    returns_risky = returns.drop(columns=[rf_ticker], errors="ignore")

    r_class, _ = get_class_returns_and_weights(
        returns_risky, weights_gmp, ticker_to_gmp
    )

    sigs = build_signals(returns_risky, include_carry=True, include_value=True)
    log.info(f"Signals: {list(sigs.keys())}")

    configs = [
        ("default_omega", "default", "default"),
        ("calib_omega", "calibrated", "default"),
        ("calib_omega_and_q", "calibrated", "calibrated"),
    ]

    results = {}
    for name, om, qm in configs:
        log.info(f"{name} (omega={om}, q={qm})")
        port, _ = bl_walk_forward(
            r_class, weights_gmp,
            window=60, rebalance=6,
            dynamic_views=True,
            tau=0.05, max_weight=0.3,
            external_signals=sigs,
            ensemble_mode="mean",
            omega_mode=om,
            omega_lookback=60,
            omega_cap_multiplier=5.0,
            q_mode=qm,
            name=name,
        )
        results[name] = port

    rows = [summary(r, n, rf_series=rf_simple) for n, r in results.items()]
    perf = pd.DataFrame(rows).set_index("name")
    log.info(f"\nOMEGA COMPARISON\n{perf.round(4).to_string()}")
    perf.to_csv(project_root() / "outputs/tables/omega_comparison_v2.csv")

    print("\n" + "=" * 70)
    print("OMEGA COMPARISON")
    print("=" * 70)
    print(perf.round(4).to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()