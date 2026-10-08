# Portfolio Management

Quantitative portfolio management research project.

This repository documents a two-stage research path: from classical mean-variance optimization (Markowitz) to a Global Market Portfolio (GMP) with Black-Litterman views. The evolution was driven by a simple insight: **the GMP is the empirical replication of the market beta**, and a beta-anchored prior is more robust out-of-sample than an unconstrained optimization on noisy estimates.

## Research narrative

### Stage 1 — Markowitz mean-variance optimization

The starting point was the classical Markowitz framework applied to a multi-asset universe:

- Max Sharpe, Min Variance, and Equal Weight strategies
- Efficient frontier, capital allocation line, walk-forward backtest
- Ledoit-Wolf shrinkage extension
- Diagnostics: weight instability, estimation error, turnover

**Finding**: mean-variance optimization is technically correct but practically fragile. The Max Sharpe portfolio is concentrated in a few assets, relies on noisy estimates of expected returns, and shows poor out-of-sample stability.

Full details: [`MarkowitzPortfolioManagement/markowitz-stage/`](./MarkowitzPortfolioManagement/markowitz-stage/).

### Stage 2 — GMP + Black-Litterman + Omega calibration

From the Markowitz critique, the project evolved into a **Global Market Portfolio** framework:

- Market-cap weighted GMP as the equilibrium prior (the empirical counterpart of market beta)
- Reverse optimization for implied equilibrium returns
- Black-Litterman posterior combining the prior with dynamic views (growth, recession, inflation)
- Point-in-time OLS calibration of Q and Ω (views)
- Walk-forward backtest with transaction costs

**Finding**: GMP + Black-Litterman produces a more stable risk-adjusted profile than pure optimization, with a plateau around Sharpe 0.69 in a long-only multi-asset setting. Omega calibration (G7) adds only +0.0017 Sharpe: the binding constraints are structural (small τ, long-only, low signal R²), not parametric.

Full details: [`GMPManagement/`](./GMPManagement/).

## Why GMP instead of Markowitz

Under the CAPM, the market portfolio is mean-variance efficient. In practice, the GMP is its empirical proxy: a market-cap weighted allocation across the investable universe. Using it as a prior in Black-Litterman:

1. Anchors the portfolio to a theoretically efficient benchmark
2. Reduces dependence on noisy expected-return estimates
3. Makes the impact of active views measurable and bounded (via τ and Ω)
4. Produces a more stable and interpretable allocation

This is why the second stage abandons unconstrained Markowitz optimization in favor of a beta-anchored, view-adjusted allocation.

## Theoretical references

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Sharpe, W. F. (1964). *Capital Asset Prices: A Theory of Market Equilibrium*. Journal of Finance.
- Black, F., & Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal.
- Idzorek, T. (2004). *A Step-by-Step Guide to the Black-Litterman Model*.
- Ledoit, O., & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix*. Journal of Portfolio Management.
- Meucci, A. (2010). *Black-Litterman Approach*. In Encyclopedia of Quantitative Finance.

## Repository structure

```text
PortfolioManagement/
├── GMPManagement/                          # Stage 2 — GMP + Black-Litterman + Omega
└── MarkowitzPortfolioManagement/           # Stage 1 — Markowitz mean-variance
    └── markowitz-stage/
```

## Shared universe

Both stages use the same risky universe for comparability:

| Ticker | Role |
|---|---|
| ACWI | Global equity |
| AGG | US aggregate bonds |
| TLT | US long-term Treasury |
| LQD | Investment grade credit |
| HYG | High yield |
| VNQ | US real estate |
| DBC | Commodities |
| GLD | Gold |
| BIL | Risk-free proxy (excluded from optimization) |

## How to run

Each stage is self-contained. See the dedicated READMEs:

- [`GMPManagement/README.md`](./GMPManagement/README.md)
- [`MarkowitzPortfolioManagement/markowitz-stage/README.md`](./MarkowitzPortfolioManagement/markowitz-stage/README.md)

## Author

Stefano Colazzilli

## License

See [LICENSE](./GMPManagement/LICENSE) for details.
