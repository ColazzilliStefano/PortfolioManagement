"""GMP analytics: risk decomposition, contribution to return, per year,
rolling correlations, drawdown table."""

import numpy as np
import pandas as pd

from src.analytics.returns import log_to_simple


# Internal utilities
def _aggregate_to_gmp_classes(returns: pd.DataFrame,
                              ticker_to_gmp: dict) -> pd.DataFrame:
    mapping = {c: ticker_to_gmp.get(c, c) for c in returns.columns}
    return returns.T.groupby(mapping).mean().T


def _align_indices(returns: pd.DataFrame,
                   weights_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = returns.copy()
    w = weights_history.copy()
    r.index = pd.to_datetime(r.index).tz_localize(None).astype("datetime64[ns]")
    w.index = pd.to_datetime(w.index, utc=True).tz_localize(None).astype("datetime64[ns]")
    return r, w


# Data preparation

def get_class_returns_and_weights(returns: pd.DataFrame,
                                  weights_history: pd.DataFrame,
                                  ticker_to_gmp: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    r, w = _align_indices(returns, weights_history)
    r_class = _aggregate_to_gmp_classes(r, ticker_to_gmp)
    w_monthly = w.reindex(r_class.index, method="ffill")
    w_monthly = w_monthly.reindex(columns=r_class.columns).fillna(0.0)
    return r_class, w_monthly


def portfolio_returns(returns_class: pd.DataFrame,
                      weights_monthly: pd.DataFrame,
                      return_type: str = "log") -> pd.Series:
    r_simple = log_to_simple(returns_class) if return_type == "log" else returns_class
    port = (r_simple * weights_monthly).sum(axis=1)
    port.name = "GMP"
    return port


# Risk decomposition

def risk_contribution(returns_class: pd.DataFrame,
                      weights_monthly: pd.DataFrame,
                      freq: int = 12,
                      lookback: int = 60,
                      min_obs: int = 24) -> pd.DataFrame:
    classes = weights_monthly.columns
    records = []

    for date in weights_monthly.index:
        w = weights_monthly.loc[date].values.astype(float)
        hist = returns_class.loc[:date].tail(lookback)
        if len(hist) < min_obs:
            continue

        cov = hist.cov().values * freq
        w = np.nan_to_num(w, nan=0.0)
        if w.sum() == 0:
            continue

        var = w @ cov @ w
        if var <= 0:
            continue
        sigma_p = np.sqrt(var)

        mcr = (cov @ w) / sigma_p
        cr = w * mcr
        pct_cr = cr / sigma_p

        records.append(pd.Series({
            "date": date,
            **{f"w_{c}": w[i] for i, c in enumerate(classes)},
            **{f"mcr_{c}": mcr[i] for i, c in enumerate(classes)},
            **{f"cr_{c}": cr[i] for i, c in enumerate(classes)},
            **{f"pct_cr_{c}": pct_cr[i] for i, c in enumerate(classes)},
            "sigma_p": sigma_p,
        }))

    if not records:
        return pd.DataFrame(columns=["weight_avg", "mcr_avg", "cr_avg", "pct_cr_avg"])

    df = pd.DataFrame(records).set_index("date")

    result = pd.DataFrame({
        "weight_avg": [df[f"w_{c}"].mean() for c in classes],
        "mcr_avg": [df[f"mcr_{c}"].mean() for c in classes],
        "cr_avg": [df[f"cr_{c}"].mean() for c in classes],
        "pct_cr_avg": [df[f"pct_cr_{c}"].mean() for c in classes],
    }, index=classes)

    return result


# Contribution to return

def contribution_to_return(returns_class: pd.DataFrame,
                           weights_monthly: pd.DataFrame) -> pd.DataFrame:
    r_simple = log_to_simple(returns_class)
    contrib = r_simple * weights_monthly
    total = contrib.mean().sum()
    return pd.DataFrame({
        "contrib_avg_monthly": contrib.mean(),
        "contrib_annualized": contrib.mean() * 12,
        "contrib_pct": contrib.mean() / total if total != 0 else np.nan,
    })


# Performance per year

def performance_by_year(port: pd.Series) -> pd.DataFrame:
    yearly = port.groupby(port.index.year).apply(lambda x: (1 + x).prod() - 1)
    return yearly.to_frame("return")


def performance_by_year_multi(curves: dict) -> pd.DataFrame:
    out = {}
    for name, r in curves.items():
        out[name] = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    return pd.DataFrame(out)


# Rolling correlations

def rolling_correlation(returns_class: pd.DataFrame, window: int = 36) -> pd.Series:
    dates = returns_class.index
    values = []

    for i in range(len(dates)):
        if i < window - 1:
            values.append(np.nan)
            continue

        sub = returns_class.iloc[i - window + 1:i + 1]
        c = sub.corr().values
        n = c.shape[0]
        if n < 2:
            values.append(np.nan)
            continue

        iu = np.triu_indices(n, k=1)
        values.append(float(np.nanmean(c[iu])))

    return pd.Series(values, index=dates, name="avg_pairwise_corr")


# Drawdown table

def drawdown_table(port: pd.Series, top: int = 5) -> pd.DataFrame:
    eq = (1 + port.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1

    events = []
    in_dd = False
    start = None

    for date, val in dd.items():
        if not in_dd and val < 0:
            in_dd = True
            start = date
        elif in_dd and val >= -1e-9:
            seg = dd.loc[start:date]
            trough = seg.idxmin()
            events.append({
                "start": start,
                "trough": trough,
                "recovery": date,
                "depth": float(seg.min()),
                "duration_days": (date - start).days,
            })
            in_dd = False

    if in_dd:
        seg = dd.loc[start:]
        trough = seg.idxmin()
        events.append({
            "start": start,
            "trough": trough,
            "recovery": None,
            "depth": float(seg.min()),
            "duration_days": (dd.index[-1] - start).days,
        })

    if not events:
        return pd.DataFrame(columns=["start", "trough", "recovery",
                                     "depth", "duration_days"])

    df = pd.DataFrame(events).sort_values("depth").head(top)
    return df.reset_index(drop=True)


# Rolling volatility
def rolling_volatility(port: pd.Series, window: int = 36, freq: int = 12) -> pd.Series:
    return port.rolling(window).std() * np.sqrt(freq)


# Rolling Sharpe with variable RF
def rolling_sharpe(port: pd.Series, rf_series: pd.Series | None = None,
                   window: int = 36, freq: int = 12) -> pd.Series:
    if rf_series is None:
        rf = pd.Series(0.0, index=port.index)
    else:
        rf = rf_series.reindex(port.index).fillna(0.0)

    ex = port - rf
    return (ex.rolling(window).mean() / port.rolling(window).std()) * np.sqrt(freq)