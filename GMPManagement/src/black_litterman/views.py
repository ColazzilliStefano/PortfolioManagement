"""Construction of the views for Black-Litterman."""

import numpy as np
import pandas as pd


def _P_row(classes: list[str], long: dict, short: dict) -> np.ndarray:
    row = np.zeros(len(classes))
    for k, v in long.items():
        if k in classes:
            row[classes.index(k)] = v
    for k, v in short.items():
        if k in classes:
            row[classes.index(k)] = -v
    return row


def build_views(classes: list[str], scenario: str,
                magnitude: float = 0.02):
    if scenario == "recession":
        long = {"bonds": 1.0}
        short = {"equity": 1.0}
    elif scenario == "growth":
        long = {"equity": 1.0}
        short = {"bonds": 1.0}
    elif scenario == "inflation_up":
        long = {"gold": 0.5, "commodities": 0.5}
        short = {"bonds": 1.0}
    elif scenario == "inflation_down":
        long = {"bonds": 1.0}
        short = {"gold": 0.5, "commodities": 0.5}
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    P = _P_row(classes, long, short).reshape(1, -1)
    Q = np.array([magnitude])
    return P, Q


def build_multi_views(classes: list[str],
                      scenarios: list[tuple[str, float]]):
    P_rows, Q_vals = [], []
    for name, mag in scenarios:
        P, Q = build_views(classes, name, mag)
        P_rows.append(P[0])
        Q_vals.append(Q[0])
    return np.array(P_rows), np.array(Q_vals)