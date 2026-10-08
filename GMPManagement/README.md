# Quantitative Portfolio Management Research

Dynamic **Global Market Portfolio (GMP)** with dynamic rebalancing based on
quantitative **Black-Litterman**, HMM regime detection, point-in-time macro
signals, and transaction costs.

This repository implements a complete research pipeline: from the critique of
classic Markowitz to a dynamic scenario-based optimizer, using the GMP as an
equilibrium benchmark and Black-Litterman to incorporate macro views.

The research path is:

```text
Markowitz
    ->
Global Market Portfolio (GMP)
    ->
Reverse optimization
    ->
Black-Litterman
    ->
Quantitative signals
    ->
Dynamic BL
```

The GMP is the equilibrium benchmark, while Dynamic BL adds Black-Litterman
views and point-in-time quantitative signals. In the current out-of-sample
backtest, the Sharpe ratio increases from **0.49 for the GMP** to **0.64 for
Dynamic BL**.

The **0.64 Sharpe** is the canonical result for the current release. The
**0.69 Sharpe** values reported in the research-path and signal-comparison
sections are historical exploratory results from earlier configurations and
are retained to document the research process.

The backtest applies point-in-time controls to CPI vintages from ALFRED,
walk-forward HMM fits, and lag-1 GMP weights. Some free-data macro and market-cap
series remain subject to the limitations documented in the
[Project limitations](#project-limitations) section.
Transaction costs are included in the final metrics.

The active signal configuration is **config-driven** (`config/config.yaml`):
signals can be selected without changing Python code.

---

## Contents

1. [Main results](#main-results)
2. [Signal configuration](#signal-configuration)
3. [Research path](#research-path)
4. [Advanced benchmarks](#advanced-benchmarks)
5. [Charts and figures](#charts-and-figures)
6. [Project architecture](#project-architecture)
7. [Methodology](#methodology)
8. [Signals and views](#signals-and-views)
9. [Omega calibration and IR weighting](#omega-calibration-and-ir-weighting)
10. [Limitations of free data sources](#limitations-of-free-data-sources)
11. [How to reproduce](#how-to-reproduce)
12. [Output policy](#output-policy)
13. [Execution notes](#execution-notes)
14. [Methodological choices](#methodological-choices)
15. [Academic references](#academic-references)
16. [Project limitations](#project-limitations)
17. [Future work](#future-work)
18. [License and disclaimer](#license-and-disclaimer)
19. [AI assistance](#ai-assistance)
20. [Changelog](#changelog)

---

## Main results

Out-of-sample backtest, 2013-2026 (**222 months**, USD risk, Sharpe with a
time-varying risk-free rate). The main table is **net**: linear transaction
costs are applied to every strategy. The gross version is saved in
`outputs/tables/final_comparison_gross.csv`.

| Strategy | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| **Dynamic BL** | **7.18%** | 8.82% | **0.64** | **0.93** | **-14.56%** | **0.49** |
| Fixed BL | 7.12% | 9.11% | 0.62 | 0.89 | -16.56% | 0.43 |
| 60/40 | 7.32% | 9.23% | 0.63 | 0.90 | -21.24% | 0.34 |
| GMP | 5.51% | 8.24% | 0.49 | 0.68 | -21.94% | 0.25 |
| Equal Weight | 4.74% | 7.96% | 0.41 | 0.59 | -16.81% | 0.28 |
| All Weather | 4.66% | 8.84% | 0.37 | 0.54 | -23.16% | 0.20 |
| Markowitz Max Sharpe | 5.87% | 7.52% | 0.58 | 0.79 | -18.02% | 0.33 |
| Markowitz Min Variance | 3.54% | 6.12% | 0.33 | 0.43 | -16.74% | 0.21 |

**Key result**: dynamic BL beats fixed BL on Sharpe (+0.02), MaxDD (+2.00 pp)
and Calmar (+0.06). It beats GMP by **+0.15 Sharpe** and **+7.38 pp MaxDD**.
The advantage is modest and has not yet been subjected to a formal statistical
test. It is obtained without look-ahead bias.

### Dynamic BL versus GMP

| Metric | Dynamic BL | GMP | Delta |
|---|---:|---:|---:|
| CAGR | 7.18% | 5.51% | **+1.67 pp** |
| Vol | 8.82% | 8.24% | +0.58 pp |
| Sharpe | 0.64 | 0.49 | **+0.15** |
| Sortino | 0.93 | 0.68 | **+0.25** |
| MaxDD | -14.56% | -21.94% | **+7.38 pp** |
| Calmar | 0.49 | 0.25 | **+0.24** |

### Look-ahead bias controls

Four corrections were applied to remove residual look-ahead bias:

1. **CPI vintage (ALFRED)**: CPI is loaded as it was available at the time,
   rather than in its revised version.
2. **Walk-forward HMM**: the regime model is fitted only on past data at each
   rebalance date and is precomputed once.
3. **Lag-1 GMP weights**: reverse optimization uses the last GMP weights with
   a date less than or equal to `dates[i-1]`, not `dates[i]`.
4. **Signal timing**: the implemented signals use information available at the
  rebalance date where vintage data is available; remaining limitations are
  documented in the [Project limitations](#project-limitations) section.

With these corrections, dynamic BL maintains a modest advantage over fixed BL.
The current version is methodologically rigorous but less spectacular than
versions using revised data.

---

## Signal configuration

Active signals are controlled by `config/config.yaml`. The
`signals.active` section determines which signals are loaded:

```yaml
signals:
  active:
    - momentum
    - hmm
    - credit
    - vix
    - term_premium
    # - carry       # disabled: does not improve Sharpe
    # - trend       # disabled: does not improve Sharpe
    # - value       # disabled: does not improve Sharpe
```

**To enable a signal**: remove the comment (`#`) from its line.
**To disable it**: add `#` at the beginning of the line.

The module `src/signals/engine.py` reads the list and loads only active signals.
**No Python code changes are required.**

### Tested configurations

The table below records exploratory v1.3 configurations from the previous
research run. The canonical current release metrics are the net values in the
main table above, generated by `run_13_final_comparison.py`.

| Configuration | Active signals | Sharpe | MaxDD | Turnover |
|---|---|---:|---:|---:|
| **Default (minimal)** | momentum, hmm, credit, vix, term_premium | **0.69** | -14.56% | 11.56% |
| Complete (6 signals) | + carry, trend, value | 0.69 | -14.45% | 12.74% |
| Omega calibrated | minimal + calibrated Omega | 0.69 | -14.61% | 11.6% |
| IR weighted | minimal + IR weighting | 0.67 | -14.38% | 11.5% |

These historical comparisons are retained to document the research process;
they are not a replacement for the current release table.

### Black-Litterman parameters

Black-Litterman parameters are also config-driven:

```yaml
bl:
  window: 60                # training months
  rebalance: 6              # months between rebalances
  tau: 0.05                 # prior uncertainty
  market_sharpe: 0.4        # expected market Sharpe
  cov_method: ledoit_wolf
  max_weight: 0.30
  long_only: true
  ensemble_mode: mean       # mean | ir_weighted
  omega_mode: default       # default | calibrated
  q_mode: default           # default | calibrated
```

---

## Research path

The project explored several configurations within a long-only multi-asset
architecture using free data. The figures below are historical research-run
results; use the main table above for the current release output:

| Configuration | Sharpe | MaxDD | Calmar | Notes |
|---|---:|---:|---:|---|
| Fixed BL (static views) | 0.65 | -16.56% | 0.45 | Baseline |
| Dynamic BL (momentum + HMM) | 0.64 | -18.34% | 0.40 | First iteration |
| **Dynamic BL (5 signals, 1/N average)** | **0.69** | **-14.56%** | **0.52** | **Historical v1.3 exploratory result** |
| Dynamic BL (6 signals) | 0.69 | -14.45% | 0.52 | carry/trend/value redundant |
| Dynamic BL + IR weighting | 0.67 | -14.38% | 0.52 | Negative test |
| Dynamic BL + calibrated Omega | 0.69 | -14.61% | 0.52 | Marginal test |

**Historical conclusion**: the plateau was approximately **Sharpe 0.69**. Every refinement
(signal selection, IR weighting, Omega calibration) changes Sharpe by only about
+/-0.02, which is statistically negligible. See
[`docs/OMEGA_CALIBRATION_REPORT.md`](docs/OMEGA_CALIBRATION_REPORT.md) for the
seven reasons behind the plateau.

---

## Advanced benchmarks

Three alternative strategies were implemented and tested as benchmarks for
dynamic BL. The following values are the current net results from
`outputs/tables/advanced_benchmarks.csv`:

| Strategy | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---:|---:|---:|---:|---:|
| **Dynamic BL** | **7.18%** | 8.82% | **0.64** | **-14.56%** | **0.49** |
| GMP | 5.52% | 8.24% | 0.49 | -21.94% | 0.25 |
| Risk Parity | 4.26% | 6.87% | 0.40 | -16.63% | 0.26 |
| HRP | 3.40% | 5.86% | 0.32 | -16.99% | 0.20 |
| CVaR | 2.05% | 5.35% | 0.09 | -14.78% | 0.14 |

**No advanced strategy beats dynamic BL** on this universe: eight
heterogeneous asset classes, unlevered, 2013-2026. Reasons:

- **Risk Parity**: unlevered and defensive, penalized by the 2022 regime
  (bonds -13%). Bridgewater uses 2-3x leverage.
- **HRP** (Lopez de Prado 2016): designed for N > 100 homogeneous assets. On
  eight heterogeneous asset classes, clustering is trivial.
- **CVaR** (Rockafellar-Uryasev 2000): minimizes downside without a return
  target, resulting in an ultra-conservative portfolio (85% bonds). The tail is
  estimated over 60 months (three observations), so the estimate is noisy.

**Interpretation**: in this context, the universe and constraints matter more
than the allocation algorithm. HRP is particularly designed for equity
portfolios with N > 100.

Script: `scripts/run_15_advanced_benchmarks.py`.

---

## Charts and figures

All figures are stored in `outputs/figures/`. Images are embedded directly in
this README using relative paths.

### Final equity curve - all strategies

![Final equity curve](outputs/figures/final_equity.png)

### Comparative drawdowns

![Final drawdowns](outputs/figures/final_drawdowns.png)

### 24-month rolling Sharpe

![Rolling Sharpe](outputs/figures/final_rolling_sharpe.png)

### Annual returns by strategy

![Annual returns](outputs/figures/final_yearly_returns.png)

### Advanced strategy comparison

![Advanced comparison](outputs/figures/advanced_equity.png)

![Advanced drawdowns](outputs/figures/advanced_drawdowns.png)

![Advanced rolling Sharpe](outputs/figures/advanced_rolling_sharpe.png)

### Annual returns table

| Year | Dynamic BL | Fixed BL | GMP | EW | 60/40 | All Weather |
|---|---:|---:|---:|---:|---:|---:|
| 2013 | -4.36% | -4.17% | +0.15% | -4.35% | +8.39% | -2.94% |
| 2014 | +8.49% | +5.52% | +8.08% | +4.86% | +4.80% | +10.16% |
| 2015 | -1.98% | -4.10% | -2.19% | -5.56% | -0.92% | -4.08% |
| 2016 | +8.94% | +9.28% | +6.86% | +8.78% | +6.12% | +6.33% |
| 2017 | +12.06% | +12.19% | +11.88% | +9.04% | +15.61% | +13.23% |
| 2018 | -5.35% | -5.34% | -4.20% | -4.15% | -5.21% | -4.64% |
| 2019 | +20.02% | +19.85% | +18.19% | +17.68% | +19.31% | +19.03% |
| 2020 | +12.55% | +10.67% | +14.47% | +9.33% | +13.47% | +16.27% |
| 2021 | +9.72% | +11.97% | +5.91% | +10.66% | +10.17% | +5.85% |
| 2022 | **-8.97%** | -11.21% | -17.59% | -12.95% | -16.04% | -19.88% |
| 2023 | +10.53% | +11.39% | +12.69% | +8.95% | +15.46% | +9.67% |
| 2024 | +12.55% | +10.94% | +7.52% | +6.53% | +10.79% | +3.85% |
| 2025 | +22.62% | +21.69% | +15.70% | +14.75% | +16.21% | +14.39% |
| 2026 | +5.40% | +13.34% | +2.79% | +5.09% | +6.54% | +2.77% |

**Key years:**

- **2022** (inflation and rates): dynamic BL -8.94% vs GMP -17.59% vs All
  Weather -19.88%.
- **2019** (recovery): dynamic BL +20.05%.
- **2025** (bull market): dynamic BL +22.65%.

### GMP - risk decomposition

| Asset class | Average weight | Risk contribution |
|---|---:|---:|
| Bonds | 61.65% | 40.44% |
| Equity | 32.18% | **53.21%** |
| Gold | 4.98% | 4.22% |
| Real estate | 0.94% | 1.88% |
| Commodities | 0.25% | 0.24% |

![GMP risk contribution](outputs/figures/gmp_risk_contribution.png)

**Equity is 32% of capital but 53% of risk.** This is the technical reason
for moving to Black-Litterman.

### GMP - return contribution

| Asset class | Annualized contribution | Share of total |
|---|---:|---:|
| Equity | 2.67% | 49.03% |
| Bonds | 2.50% | 45.89% |
| Gold | 0.46% | 8.38% |
| Real estate | -0.09% | -1.57% |
| Commodities | -0.09% | -1.73% |

![GMP return contribution](outputs/figures/gmp_return_contribution.png)

### GMP - top five drawdowns

| # | Start | Trough | Recovery | Depth | Duration |
|---|---|---|---|---:|---:|
| 1 | 2008-05-31 | 2009-02-28 | 2011-10-31 | **-32.65%** | 1248 days |
| 2 | 2022-01-31 | 2022-09-30 | 2024-08-31 | **-21.94%** | 943 days |
| 3 | 2020-02-29 | 2020-03-31 | 2020-05-31 | -7.22% | 92 days |
| 4 | 2013-05-31 | 2013-06-30 | 2014-02-28 | -5.92% | 273 days |
| 5 | 2015-03-31 | 2015-09-30 | 2016-06-30 | -5.69% | 457 days |

### BL - turnover and costs

| Strategy | Average turnover | Maximum turnover | Cost drag |
|---|---:|---:|---:|
| Dynamic BL | 11.56% | 40.39% | ~4 bps/year |
| Fixed BL | 4.71% | 27.31% | ~1 bps/year |
| GMP | - | - | ~1 bps/year |

The estimated cost drag remains contained because of semiannual rebalancing.
The CAGR drag ranges from approximately 0.5 bps/year for 60/40 to approximately
7.0 bps/year for Markowitz Max Sharpe. Full details are in
`outputs/tables/costs_impact_final.csv`; turnover for all strategies is in
`outputs/tables/turnover_comparison.csv`.

### Pi - implied returns from reverse optimization

| Asset | Pi |
|---|---:|
| Real estate | 6.43% |
| Equity | 5.97% |
| Commodities | 2.95% |
| Gold | 2.42% |
| Bonds | 2.19% |

Risk aversion lambda = **4.31**.

### Phase 1 - Markowitz critique

**Efficient frontier:**

![Constrained Markowitz frontier](outputs/figures/markowitz/frontier_markowitz_constrained.png)

**Capital Allocation Line:**

![Capital Allocation Line](outputs/figures/markowitz/capital_allocation_line_constrained.png)

Both figures use the same monthly return sample, annualized expected returns
and covariance matrix, risk-free rate, long-only constraint, and 30% maximum
position size. The Capital Allocation Line starts at the risk-free asset and
passes through the constrained maximum-Sharpe portfolio shown on the
efficient frontier. The remaining Markowitz figures use the same 60-month
walk-forward estimation framework and the same allocation constraints.

**Max Sharpe weights (in sample):**

![Max Sharpe weights](outputs/figures/markowitz/weights_tangency.png)

**Minimum variance weights (in sample):**

![Minimum variance weights](outputs/figures/markowitz/weights_minvar.png)

**Maximum Sharpe weight instability:**

![Weight instability](outputs/figures/markowitz/weights_instability.png)

**Estimation error (in-sample Sharpe vs out-of-sample):**

![Estimation error](outputs/figures/markowitz/estimation_error.png)

**Markowitz equity curve:**

![Markowitz equity](outputs/figures/markowitz/equity_curves.png)

**Markowitz drawdown:**

![Markowitz drawdown](outputs/figures/markowitz/drawdowns.png)

**Correlation heatmap:**

![Correlation heatmap](outputs/figures/markowitz/correlation_heatmap.png)

**Turnover:**

![Turnover](outputs/figures/markowitz/turnover.png)

### Phase 2 - Global Market Portfolio

![GMP weights](outputs/figures/gmp_weights_evolution.png)

![GMP equity](outputs/figures/gmp_equity_curve.png)

![GMP drawdown](outputs/figures/gmp_drawdowns.png)

![GMP rolling Sharpe](outputs/figures/gmp_rolling_sharpe.png)

![GMP rolling correlation](outputs/figures/gmp_rolling_correlation.png)

![GMP turnover](outputs/figures/gmp_turnover.png)

![GMP versus benchmarks](outputs/figures/gmp_vs_benchmarks.png)

![GMP annual returns](outputs/figures/gmp_yearly_returns.png)

### Phases 3-5 - Black-Litterman

![BL versus GMP](outputs/figures/bl_vs_gmp_equity.png)

![BL versus GMP drawdown](outputs/figures/bl_vs_gmp_drawdowns.png)

![BL versus GMP rolling Sharpe](outputs/figures/bl_vs_gmp_rolling_sharpe.png)

![BL weights](outputs/figures/bl_dynamic_weights.png)

![BL turnover](outputs/figures/bl_turnover.png)

![Dynamic BL equity](outputs/figures/bl_dynamic_equity.png)

![Dynamic BL drawdown](outputs/figures/bl_dynamic_drawdowns.png)

![Dynamic BL rolling Sharpe](outputs/figures/bl_dynamic_rolling_sharpe.png)

![Dynamic BL weights](outputs/figures/bl_dynamic_weights.png)

### Transaction costs

![Transaction-cost impact](outputs/figures/costs_impact_equity.png)

![Transaction-cost drawdowns](outputs/figures/costs_impact_drawdowns.png)

![Transaction-cost rolling Sharpe](outputs/figures/costs_impact_rolling_sharpe.png)

### Opening the figures

**From PowerShell** (project root):

```powershell
start outputs\figures
start outputs\figures\final_equity.png
```

**Open all figures in sequence:**

```powershell
gci outputs\figures\*.png | ForEach-Object { start $_.FullName; Start-Sleep -Seconds 1 }
```

**From VS Code**: Explorer panel -> `outputs/figures/` -> click a PNG.

**Markdown preview in VS Code**: open `README.md` and press `Ctrl+Shift+V`.

### Recommended reading order

1. `gmp_weights_evolution.png` - GMP weights over time.
2. `gmp_risk_contribution.png` - equity is 32% of capital and 53% of risk.
3. `final_equity.png` - all strategies compared.
4. `final_drawdowns.png` - which strategies suffer most in crises.
5. `final_rolling_sharpe.png` - Sharpe stability.
6. `advanced_equity.png` - Risk Parity/HRP/CVaR versus BL.
7. `bl_dynamic_weights.png` - how views change portfolio weights.
8. `costs_impact_equity.png` - transaction-cost impact.

---

## Project architecture

```text
GMPManagement/
|- config/                  # YAML configuration
|  |- config.yaml           # Master: signals.active, BL parameters, benchmarks
|  |- assets.yaml           # Asset universe and ticker_to_gmp
|  |- benchmarks.yaml       # Benchmark parameters
|  `- gmp_dynamic.yaml      # GMP configuration
|- data/                    # Raw and processed parquet data
|  |- raw/
|  |  |- prices/
|  |  |- market_caps/
|  |  `- macro/             # ALFRED vintages, Shiller, and VIX
|  |- interim/
|  `- processed/
|     |- returns_monthly.parquet
|     |- gmp_weights_history.parquet
|     |- gmp_returns.parquet
|     |- bl_returns.parquet
|     |- bl_dyn_returns.parquet
|     `- data_provenance.yaml
|- docs/
|  `- OMEGA_CALIBRATION_REPORT.md
|- src/
|  |- data/                # Download, cleaning, and provenance
|  |  |- fred_client.py    # FRED/ALFRED client with API key
|  |  |- downloader.py
|  |  |- cleaner.py
|  |  `- provenance.py
|  |- market_caps/         # Institutional-source parsers
|  |- gmp/                 # GMP construction and analytics
|  |- benchmarks/          # EW, 60/40, All Weather, Risk Parity, and HRP
|  |  |- equal_weight.py
|  |  |- sixty_forty.py
|  |  |- all_weather.py
|  |  |- risk_parity.py    # Equal Risk Contribution
|  |  `- hrp.py             # Hierarchical Risk Parity
|  |- risk/                # Alternative risk measures
|  |  `- cvar.py            # CVaR optimization (Rockafellar-Uryasev)
|  |- estimators/          # Covariance and mean shrinkage
|  |- black_litterman/     # Reverse optimization, BL, and walk-forward
|  |  |- bl_model.py
|  |  |- bl_backtest.py
|  |  |- reverse_optimization.py
|  |  |- views.py
|  |  `- omega_calibration.py # OLS Omega calibration
|  |- signals/             # Signals (10 modules)
|  |  |- engine.py         # Centralized, config-driven signal engine
|  |  |- fred_loader.py
|  |  |- credit.py
|  |  |- vix.py
|  |  |- inflation.py      # CPI and term premium
|  |  |- yield_curve.py
|  |  |- momentum.py
|  |  |- regimes.py        # HMM
|  |  |- carry.py
|  |  |- trend.py
|  |  |- value.py
|  |  |- shiller.py        # CAPE loader
|  |  `- ensemble.py       # IR weighting
|  |- costs/               # Transaction costs
|  |  `- transaction_costs.py
|  |- markowitz/           # Markowitz modules
|  |  |- mean_variance.py
|  |  |- efficient_frontier.py
|  |  |- backtest.py
|  |  |- critique.py
|  |  `- shrinkage.py
|  |- analytics/           # Returns, performance, and statistics
|  |- viz/                 # Charts
|  `- utils/               # Configuration, logging, and manifest
|     |- config.py
|     |- logger.py
|     `- manifest.py
|- scripts/                # Executable entry points
|  |- run_01_download.py
|  |- run_02_clean.py
|  |- run_03_validate.py
|  |- seed_market_caps.py
|  |- run_04_market_caps.py
|  |- run_05_gmp_weights.py
|  |- run_06_gmp_backtest.py
|  |- run_07_gmp_vs_benchmarks.py
|  |- run_08_gmp_analytics.py
|  |- run_09_reverse_opt.py
|  |- run_10_black_litterman.py
|  |- run_11_bl_backtest.py
|  |- run_12_bl_dynamic.py
|  |- run_12b_ensemble_comparison.py
|  |- run_12c_omega_comparison.py
|  |- run_13_final_comparison.py
|  |- run_14_costs_impact.py
|  `- run_15_advanced_benchmarks.py
|- outputs/
|  |- figures/             # PNG figures
|  |- tables/              # CSV results
|  |- weights/             # Historical weights
|  |- data_quality/        # Data-quality reports
|  `- runs/                # Execution manifests
|- tests/                  # Test suite
|- .env                    # FRED_API_KEY, not versioned
|- .env.example            # Template
|- .gitignore
|- requirements.txt
|- requirements.lock.txt
`- README.md
```

---

## Methodology

### Phase 1 - Markowitz critique

The classic mean-variance model fails out of sample for three reasons:

1. **Estimation error**: an in-sample/out-of-sample Sharpe gap of -95%.
   In-sample Sharpe 0.64 -> out-of-sample 0.03.
2. **Corner solutions**: Minimum Variance concentrates 93% in AGG. Maximum
   Sharpe assigns zero weight to four of eight assets.
3. **Instability**: perturbing mu by +/-0.1% moves 12% of AGG's weight.

The Markowitz-stage figures are embedded in the Phase 1 section above.

### Phase 2 - Global Market Portfolio

Weights from institutional sources:

| Asset class | Source | Frequency | Metric |
|---|---|---|---|
| Equity | MSCI ACWI factsheet | Annual | Aggregate market cap |
| Bonds | BIS Debt Securities | Quarterly | Debt outstanding |
| Real estate | FTSE EPRA Nareit | Annual | Listed REITs |
| Commodities | CFTC Open Interest | Monthly | Futures open interest |
| Gold | World Gold Council | Annual | Above-ground stock |

Weights are updated at a configurable frequency: monthly, semiannual, or
annual. SSGA 2025 / Doeswijk 2019 are used as fallbacks when a source is
missing.

**GMP results, 2008-2026:**

- CAGR 5.52%, volatility 8.24%, Sharpe 0.49, MaxDD -21.94%.
- Equity is 32% of capital but **53% of risk** (risk-imbalanced).
- Bonds are 62% of capital, 40% of risk, and 46% of return.

### Phase 3 - Reverse optimization and shrinkage

Equilibrium implied returns:

```text
Pi = lambda * Sigma * w_gmp
```

where:

- `lambda` is calibrated to a market Sharpe of 0.4 -> lambda = 4.31;
- `Sigma` is estimated using Ledoit-Wolf shrinkage.

**Annualized results:**

| Asset | Pi |
|---|---:|
| Real estate | 6.43% |
| Equity | 5.97% |
| Commodities | 2.95% |
| Gold | 2.42% |
| Bonds | 2.19% |

With T/N = 45, shrinkage has a negligible impact. It becomes relevant when N
approaches T.

### Phase 4 - Black-Litterman

Posterior formula:

```text
E[r]   = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
         * [(tau Sigma)^-1 Pi + P' Omega^-1 Q]

Cov[r] = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
```

with:

- prior: `Pi`, `tau = 0.05`;
- view: `P`, `Q`, `Omega = tau * P Sigma P'`;
- regularization on Omega and tau*Sigma (`ridge = 1e-8`) for robustness;
- pseudo-inverse (`pinv`) to handle collinear views.

### Phase 5 - Dynamic views (Quantitative BL)

Views are generated automatically at each rebalance. They are all
**point-in-time**:

| Signal | Source | Method | Estimated IR |
|---|---|---|---:|
| **Equity momentum** 6m | yfinance | Rolling, no look-ahead | 0.05-0.10 |
| **Three-state HMM regime** | yfinance | Walk-forward fit on equity, bonds, gold | 0.05-0.10 |
| **Credit spread** (HYG vs LQD) | yfinance | 6m differential | 0.15-0.25 |
| **VIX term structure** (^VIX vs ^VIX3M) | yfinance | Contango/backwardation | 0.20-0.35 |
| **Term premium** (10y - 3m) | FRED | Curve slope | 0.15-0.25 |
| **Carry** (CAPE ERP) | FRED + Shiller | Earnings yield - 10y | 0.10-0.20 |
| **Trend** 12m | yfinance | Long momentum | 0.20-0.30 |
| **Value** (CAPE deviation) | Shiller | Mean reversion | 0.10-0.20 |
| **Inflation** CPI YoY | **ALFRED vintage** | Point-in-time | 0.10-0.15 |
| **Yield curve** 10y-2y | FRED | Recession indicator | 0.05-0.10 |

**Walk-forward HMM**: precomputed once (27 separate fits using past data only).
The resulting series is used both as the current value and for IR calculation.

**CPI vintage**: every rebalance downloads CPI as it was available at that
time from ALFRED. This removes the residual look-ahead caused by revisions.

### Phase 6 - Optimization

Mean-variance optimization with constraints:

- long-only;
- `max_weight = 0.30`;
- weights sum to one.

```text
w* = argmax  w' mu_bl  -  0.5 * lambda * w' Sigma w
```

**Important**: `Sigma` (return covariance) is used, not `cov_bl` (uncertainty
about expected returns).

### Phase 7 - Transaction costs

Asset-class transaction-cost model in basis points:

- Equity: 5 bps
- Bonds: 10 bps
- Gold: 10 bps
- Real estate: 12 bps
- Commodities: 15 bps

Cost per rebalance = sum of `|delta weight| * bps / 10000`.

**Estimated drag**: approximately 2-4 bps/year. It remains small because of
semiannual rebalancing.

---

## Signals and views

### Generated views

| Scenario | Long | Short |
|---|---|---|
| growth | equity 1.0 | bonds 1.0 |
| recession | bonds 1.0 | equity 1.0 |
| inflation_up | gold 0.5 + commodities 0.5 | bonds 1.0 |
| inflation_down | bonds 1.0 | gold 0.5 + commodities 0.5 |

### Growth-view composition

The `growth` view is a 1/N weighted average of the following signals:

```text
growth_view = mean(momentum_6m, hmm_regime, credit, vix, term_premium,
                    [carry, trend, value])
```

Each signal is normalized to [-0.06, +0.06].

### Signal-calibrated magnitudes

```text
momentum_6m     ->  tanh(momentum / 0.10) * 0.06
HMM regime      ->  (p_bull - p_bear) * 0.06
credit spread   ->  -tanh(z_score / 1.0) * 0.06
vix term struct ->  tanh(z_contango / 1.0) * 0.05
term premium    ->  tanh(z_slope / 1.0) * 0.04
carry (ERP)     ->  tanh(z_erp / 1.0) * 0.05
trend 12m       ->  tanh(trend / 0.15) * 0.05
value (CAPE)    ->  -tanh(z_cape / 1.0) * 0.05
CPI YoY         ->  tanh((CPI - 2.0) / 1.5) * 0.04
yield curve     ->  tanh(-(spread) / 1.0) * 0.05
```

If `|magnitude| < 1e-4`, the view is excluded.

---

## Omega calibration and IR weighting

Two methodological extensions were implemented and tested. Both produce
negligible changes in Sharpe.

### Information Ratio weighted ensemble

Instead of a 1/N average, growth signals are weighted by their rolling
60-month IR:

```text
w_signal = max(0, IR_signal) / sum(max(0, IR_all_signals))
```

**Result**: Sharpe 0.67 versus 0.69 for the default, a change of **-0.02**.

**Interpretation**: IR estimated over 60 months is noisy; the 1/N average is
more robust (DeMiguel-Garlappi-Uppal 2009).

### Omega calibration

Instead of `Omega = tau * P Sigma P'`, Omega is calibrated from the residual
variance of a point-in-time OLS regression:

```text
realized(t+1) = alpha + beta * view(t) + epsilon(t)
Omega_i = var(epsilon)
```

**Result**: Sharpe 0.69 versus 0.69 for the default, a change of **+0.0004**.

**With Q calibration**: Sharpe 0.69, a change of **+0.0017**.

**Interpretation**: the calibration is dimensionally correct (calibrated Omega
= 1.3-1.8x default), but signal R-squared is below 0.03. Omega calibration
adds no value when signals have little predictive power.

See [`docs/OMEGA_CALIBRATION_REPORT.md`](docs/OMEGA_CALIBRATION_REPORT.md) for
the complete analysis of the seven reasons for the plateau.

---

## Limitations of free data sources

### yfinance

- **Market cap**: current values only, not historical values.
- **ETF AUM** is NOT market capitalization. Empirical checks showed:
  - Equity: AUM 3.2% versus actual GMP approximately 43%.
  - Bonds: AUM 61.9% versus actual GMP approximately 42%.
  - Gold: AUM 25.5% versus actual GMP approximately 5-8%.
- **Prices**: reliable for liquid ETFs, with opaque adjustments.
- **VIX term structure**: available from 2011 (VIX3M, VIX9D).

### FRED / ALFRED

- **FRED**: revised series, not point-in-time. A free API key is required for
  full access.
- **ALFRED**: point-in-time vintages, used for CPI. Latency is 1-2 seconds per
  vintage, approximately 27 vintages on the first run.
- **ICE BofA credit spread** (`BAMLH0A0HYM2`): limited to 795 observations
  (three years) through the API because of licensing restrictions. Replaced by
  an HYG/LQD proxy.

### Shiller dataset

- **CAPE**: 1,725 observations since 1881, updated monthly.
- **Download**: XLS file from Yale; requires `xlrd` for parsing.

### Institutional sources

- **MSCI ACWI**: annual update, aggregate market cap.
- **BIS**: quarterly frequency; includes non-investable debt.
- **FTSE EPRA**: listed REITs only, no private real estate.
- **CFTC**: open interest is not market capitalization.
- **WGC**: above-ground stock, updated annually.

### Expected impact

| Aspect | Free source | Paid source |
|---|---|---|
| Historical market cap | Not available | Bloomberg, Refinitiv |
| Point-in-time data | ALFRED (partial) | Bloomberg, ICE BofA |
| Frequency | Annual/quarterly | Real-time |
| Granularity | Aggregate | Sector, rating, duration |
| Cost | 0 | EUR 50k-150k/year |

---

## How to reproduce

The commands below rebuild all downloaded and derived data from scratch.
They require network access and a valid FRED API key. The exact local
environment can be installed from `requirements.lock.txt`.

### Initial setup

```powershell
git clone <repository-url>
cd GMPManagement

py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

For a deterministic environment, use the UTF-8 lockfile after creating the
virtual environment:

```powershell
pip install -r requirements.lock.txt
```

### FRED API key configuration

Create a `.env` file in the project root:

```text
FRED_API_KEY=your_32_character_key
```

Get a free key at https://fredaccount.stlouisfed.org/apikeys.

The `.env` file is excluded from version control.

### requirements.txt

```text
pandas>=2.0,<3.0
numpy>=1.24,<2.0
scipy>=1.10,<2.0
yfinance>=0.2.40
pyarrow>=14.0
pyyaml>=6.0
loguru>=0.7
matplotlib>=3.7
seaborn>=0.13
hmmlearn>=0.3
requests>=2.31
pytest>=7.4
fredapi>=0.5.1
xlrd>=2.0.2
```

### Complete pipeline

```powershell
python scripts\run_01_download.py
python scripts\run_02_clean.py
python scripts\run_03_validate.py
python scripts\seed_market_caps.py
python scripts\run_04_market_caps.py
python scripts\run_05_gmp_weights.py --frequency semiannual
python scripts\run_06_gmp_backtest.py
python scripts\run_07_gmp_vs_benchmarks.py
python scripts\run_08_gmp_analytics.py
python scripts\run_09_reverse_opt.py
python scripts\run_10_black_litterman.py
python scripts\run_11_bl_backtest.py
python scripts\run_12_bl_dynamic.py
python scripts\run_13_final_comparison.py
python scripts\run_14_costs_impact.py
python scripts\run_15_advanced_benchmarks.py
```

### Comparison scripts (G7/G8/B1/B2/B4 tests)

```powershell
python scripts\run_12b_ensemble_comparison.py    #  mean versus IR weighting
python scripts\run_12c_omega_comparison.py       #  default versus calibrated Omega
python scripts\run_15_advanced_benchmarks.py     # RP versus HRP versus CVaR
```

### Test verification

```powershell
pytest tests/ -v
```

Expected result: **68 passed**.

### Outputs

| Folder | Contents | Main files |
|---|---|---|
| `outputs/figures/` | PNG charts | 24 figures in the current snapshot |
| `outputs/tables/` | CSV tables | 39 tables in the current snapshot |
| `outputs/weights/` | Historical GMP weights | `gmp_weights_history.csv` |
| `outputs/data_quality/` | Quality reports | `fallback_report.csv`, `summary_stats.csv` |
| `outputs/runs/` | Execution manifests | `manifest.json` |
| `data/processed/` | Intermediate parquet | returns, weights, portfolio returns |

---

## Output policy

### Versioned outputs

- **Figures** in `outputs/figures/*.png`: required by the README and part of
  the visual documentation.
- **Documentation** in `docs/OMEGA_CALIBRATION_REPORT.md`: the technical report
  on Omega calibration and the Sharpe plateau.
- **Configurations** in `config/*.yaml`: pipeline reproducibility.

### Non-versioned outputs

- **CSV tables** in `outputs/tables/`: regenerable with the scripts.
- **Raw data** in `data/raw/`: downloaded automatically.
- **Processed data** in `data/processed/`: regenerable with the pipeline.
- **Manifests** in `outputs/runs/`: specific to each execution.
- **Historical weights** in `outputs/weights/`: regenerable.

### Regenerating everything

```powershell
# Complete pipeline
python scripts\run_01_download.py
python scripts\run_02_clean.py
# ... (see the "How to reproduce" section)
```

### `.env` file

Contains `FRED_API_KEY` and must **never be versioned**; it is excluded by
`.gitignore`.

---

## Execution notes

- **Dataset**: 223 monthly observations (2008-03 to 2026-09). The final
  out-of-sample comparison uses the common available evaluation window.
- **Fallbacks**: 10 events recorded, all in 2008 before institutional CSV files
  began. Full report: `outputs/data_quality/fallback_report.csv`.
- **Execution manifest**: `outputs/runs/<timestamp>/manifest.json`.
- **Data provenance**: `data/processed/data_provenance.yaml`.
- **Tests**: 68 tests passed (`pytest tests/ -v`).
- **Dependencies**: `requirements.txt` for ranges and `requirements.lock.txt`
  for exact versions.
- **ALFRED vintages**: approximately 27 CSV files are downloaded on the first
  run, one per rebalance date. Later runs use the cache.
- **FRED API**: requires `FRED_API_KEY` in `.env`. Rate limit: 120 requests per
  minute.
- **Shiller dataset**: downloaded from Yale on the first run, approximately
  1.6 MB.
- **VIX term structure**: downloaded through yfinance, available from 2011.
- **Sharpe**: calculated with a time-varying risk-free rate (BIL), consistently
  across `run_12`, `run_13`, and `run_15` using the same signal engine.

---

## Methodological choices

### Why store log returns and analyze simple returns

Log returns are additive over time and useful for storage. Analysis (mu, Sigma,
aggregation, and metrics) uses simple returns.

```python
simple = np.expm1(log_returns)
log    = np.log1p(simple_returns)

# CORRECT
port_simple = (np.expm1(log_returns) * w).sum(axis=1)
```

### Why Ledoit-Wolf instead of sample covariance

With T/N = 45, the difference is negligible. It becomes relevant when N tends
toward T.

### Why walk-forward HMM

Fitting an HMM on the full series introduces look-ahead bias. Walk-forward
fitting guarantees that the model sees only past data.

### Why CPI vintages from ALFRED

Revised CPI contains information that was not available at the time. Replacing
it with the vintage point-in-time version removes this residual look-ahead.

### Why lag-1 GMP weights

Reverse optimization uses the GMP weights from the last date less than or equal
to `dates[i-1]`, not `dates[i]`. This prevents using information from the month
being applied.

### Why HYG versus LQD as a credit-spread proxy

The FRED series `BAMLH0A0HYM2` is limited to 795 observations (three years)
through the API because of licensing restrictions. The HYG/LQD proxy tracks the
HY-IG spread movement across the full backtest period, starting in 2008.

### Why a 1/N average instead of IR weighting

IR estimated over a 60-month window is noisy. The simple 1/N average is
**robust** and difficult to beat out of sample, as documented by
DeMiguel-Garlappi-Uppal (2009). A/B tests confirm that IR weighting produces
Sharpe 0.67 versus 0.69 for the default.

### Why config-driven signals

Active signals are read from `config/config.yaml`. This makes it possible to:

- enable or disable signals without touching code;
- test configurations reproducibly;
- document the active setup by reading one file.

### Why Risk Parity, HRP, and CVaR are not competitive

Three structural reasons, rather than implementation problems:

- **Risk Parity**: unlevered. Bridgewater uses 2-3x leverage to amplify return.
  Leverage does not change Sharpe by definition, but it increases CAGR.
- **HRP**: designed for N > 100 homogeneous assets, such as the S&P 500. On
  eight heterogeneous asset classes, clustering is trivial.
- **CVaR**: minimizes downside without a return target, producing an
  ultra-conservative portfolio (85% bonds). The tail is estimated over 60
  months (three observations), so the estimate is noisy.

### Why `tau = 0.05`

A standard value in the literature (Idzorek 2004, He-Litterman 1999).

### Why `max_weight = 0.30`

It prevents extreme concentrations.

### Why transaction costs are included

The drag is approximately 2-4 bps/year because of semiannual rebalancing, but
it is included for methodological honesty.

---

## Academic references

- **Black, F., Litterman, R.** (1992). "Global Portfolio Optimization".
  *Financial Analysts Journal*, 48(5), 28-43.
- **DeMiguel, V., Garlappi, L., Uppal, R.** (2009). "Optimal Versus Naive
  Diversification". *Review of Financial Studies*, 22(5), 1915-1953.
- **Doeswijk, R., Lam, T., Swinkels, L.** (2019). "Historical Returns of the
  Market Portfolio". *Review of Asset Pricing Studies*, 10(2), 298-345.
- **Idzorek, T.** (2004). "A Step-by-Step Guide to the Black-Litterman Model".
- **Ledoit, O., Wolf, M.** (2004). "Honey, I Shrunk the Sample Covariance
  Matrix". *Journal of Portfolio Management*, 30(4), 110-119.
- **Lopez de Prado, M.** (2016). "Building Diversified Portfolios that
  Outperform Out-of-Sample". *Journal of Portfolio Management*, 42(4), 59-69.
- **Maillard, S., Roncalli, T., Teiletche, J.** (2010). "The Properties of
  Equally Weighted Risk Contribution Portfolios". *Journal of Portfolio
  Management*, 36(4), 60-70.
- **Markowitz, H.** (1952). "Portfolio Selection". *Journal of Finance*,
  7(1), 77-91.
- **Meucci, A.** (2010). "The Black-Litterman Approach: Original Model and
  Extensions". *SSRN Working Paper*.
- **Rockafellar, R.T., Uryasev, S.** (2000). "Optimization of Conditional
  Value-at-Risk". *Journal of Risk*, 2(3), 21-41.
- **Hamilton, J.D.** (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle". *Econometrica*, 57(2).
- **Ang, A., Bekaert, G.** (2002). "International Asset Allocation with Regime
  Shifts". *Review of Financial Studies*, 15(4), 1137-1187.
- **Moskowitz, T., Ooi, Y.H., Pedersen, L.H.** (2012). "Time Series Momentum".
  *Journal of Financial Economics*, 104(2), 228-250.

---

## Project limitations

1. **Universe**: eight ETF-based asset classes. Private equity, private credit,
   and direct real estate are missing.
2. **Market cap**: proxies from free sources, not true point-in-time data.
3. **HMM**: numerical instability with short series (less than 60 months).
4. **View magnitudes**: hyperparameters selected ex post.
5. **Currency**: USD base, with no currency hedge.
6. **Period**: 2008-2026, one complete market cycle.
7. **Signals**: estimated IR 0.05-0.30 and R-squared below 0.05, below the
   usefulness threshold for Omega calibration.
8. **Omega calibration**: professionally implemented (G7), but negligible
   impact (+0.0017 Sharpe).
9. **IR weighting**: implemented (G8), but lowers Sharpe (-0.02).
10. **Risk Parity**: not competitive when unlevered (Sharpe 0.44).
11. **HRP**: not competitive on eight heterogeneous assets (Sharpe 0.36); it
    is designed for N > 100.
12. **CVaR**: not competitive (Sharpe 0.11); it minimizes only downside and
    estimates the tail over 60 months.
13. **No market-impact model**: linear costs only, no Almgren-Chriss model.
14. **Long-only architecture**: prevents direct expression of negative views.
15. **Credit spread**: ICE BofA series on FRED is limited to three years.
16. **Carry, trend, value**: tested but disabled because they do not improve
    Sharpe.

**Complete plateau analysis**: see
[`docs/OMEGA_CALIBRATION_REPORT.md`](docs/OMEGA_CALIBRATION_REPORT.md).

---

## Future work

### Structural directions (high priority)

To move beyond the Sharpe ~0.69 plateau, architectural changes are required:

- **Long-short**: requires prime brokerage (EUR 50k-200k/year), removes the
  long-only constraint, and has potential Sharpe 1.0-1.5.
- **Reinforcement Learning**: replaces the view engine with an agent that learns
  the policy. Estimated development time is two or more weeks; potential Sharpe
  0.7-0.9.
- **Proprietary data**: Bloomberg or Refinitiv (EUR 50k-150k/year), potential
  Sharpe 1.2-1.5.
- **Event-driven backtest**: simulates orders, latency, and partial fills.

### Signal improvements (medium priority)

Replace current signals with alternatives having a higher documented IR:

- **BAMLH0A0HYM2 credit spread** (true series, not proxy): requires an ICE BofA
  license for full history.
- **VIX with nonlinear transformation**: potential R-squared 0.03-0.06.
- **Order flow / market microstructure**: requires tick data.
- **News sentiment**: requires a provider such as Ravenpack or GDELT.

### Extended macro vintages (medium priority)

Extend ALFRED vintages to:

- GDP and unemployment;
- PMI as a leading indicator;
- Fed funds rate;
- Federal Reserve balance sheet (QE/QT).

### Methodological extensions (medium priority)

- Mean-CVaR with an explicit return target.
- Levered Risk Parity at a volatility target (Sharpe unchanged, CAGR up).
- HRP on a broad universe (20-30 sector ETFs), as a separate project.
- Multi-period optimization (MPC) with a rolling horizon.
- Bayesian BL with a distribution on Q instead of a point estimate.

### Dashboard and operations (medium priority)

- Streamlit dashboard for current weights.
- Scheduler (Task Scheduler / cron) for automatic recalculation.
- Docker container for portable reproducibility.
- FastAPI REST service exposing weights.

### Research and publication (low priority)

- Technical blog post describing the pipeline and results.
- Academic paper analyzing the architectural plateau.
- Interactive Jupyter / Google Colab notebook.

### Hedge-fund production (low priority, long term)

- Market-impact model (Almgren-Chriss).
- Broker API (Interactive Brokers, Alpaca).
- Custom OMS/EMS.
- Real-time risk management.

---

## Technical documentation

- [`docs/OMEGA_CALIBRATION_REPORT.md`](docs/OMEGA_CALIBRATION_REPORT.md) - G7
  technical report and complete plateau analysis

---

## License and disclaimer

This repository is **not open source**. It is provided for personal research
and evaluation only under the proprietary terms in [`LICENSE`](LICENSE).
Redistribution, commercial use, and publication of modified versions require
prior written permission.

The project is not investment advice and does not constitute an offer or
recommendation to buy or sell financial instruments. Backtested results are
historical research results, not a guarantee of future performance.
Third-party data and libraries remain subject to their own licenses and terms.

## AI assistance

AI tools were used to assist with Python syntax, refactoring, documentation,
and test-oriented code review. The research design, methodological choices,
interpretation of results, and validation of the backtest remain the
responsibility of the author.

---

## Contact

Project developed for studying portfolio optimization with quantitative
Black-Litterman.

---

## Changelog

- **v1.3** (2026-09-22): Advanced benchmarks and integrity fixes.
  - **New modules**:
    - `src/benchmarks/risk_parity.py` - B1: Equal Risk Contribution
    - `src/benchmarks/hrp.py` - B2: Hierarchical Risk Parity
    - `src/risk/cvar.py` - B4: CVaR optimization (Rockafellar-Uryasev)
    - `scripts/run_15_advanced_benchmarks.py` - five-strategy comparison
    - `tests/test_advanced_benchmarks.py` - 10 tests
  - **15 fixes applied**:
    1. Config-driven `run_15` reads `advanced_benchmarks` from YAML.
    2. Costs standardized across all strategies.
    3. Added RP/HRP/CVaR advanced tests.
    4. Added API annotation `tuple[pd.Series, pd.DataFrame]`.
    5. Added `max_weight >= 1/n` validation with a consistent fallback.
    6. Saved historical weights (`advanced_*_weights.csv`).
    7. Removed hardcoded parameters from `run_15`.
    8. Updated the README pipeline.
    9. Documented B1/B2/B4 as tested but not competitive.
    10. Added a manifest for `run_15`.
    11. Added `xlrd>=2.0.2` to requirements.
    12. Added `OMEGA_CALIBRATION_REPORT.md` in uppercase.
    13. Fixed a documentation typo.
    14. Removed unused imports.
    15. Documented the output policy.
  - **Net results, 2013-2026**:
    - Dynamic BL Sharpe 0.64, MaxDD -14.56%.
    - GMP Sharpe 0.49, MaxDD -21.94%.
    - Risk Parity Sharpe 0.40 (not competitive).
    - HRP Sharpe 0.32 (designed for N > 100).
    - CVaR Sharpe 0.09 (minimizes downside only).
  - **Tests**: 65 passed at release time. The current suite has 68 passing
    tests. Compilation OK. `pip check` clean.

- **v1.2** (2026-09-20): Config-driven signals and centralized signal engine.
  - **`src/signals/engine.py`**: centralized, config-driven signal engine.
  - **`config/config.yaml`**: `signals.active` and BL parameters.
  - **`src/utils/config.py`**: `get_active_signals` and `get_bl_config` helpers.
  - **Omega calibration (G7)** with point-in-time OLS regression.
  - **IR weighting (G8)** with rolling Information Ratio.
  - **`docs/OMEGA_CALIBRATION_REPORT.md`**: analysis of the seven plateau
    reasons.
  - **`run_12` and `run_13` aligned**: same signal engine and same numbers.
  - **Results**: dynamic BL Sharpe 0.69, MaxDD -14.56%, Calmar 0.52.
  - **Minimal configuration**: carry, trend, and value disabled.
  - README updated with final results.

- **v1.1** (2026-09-20): Signal expansion.
  - **Six new signals**: credit-spread proxy, VIX term structure, term premium,
    carry, trend, and value.
  - **FRED API client** with ALFRED vintage support (`fredapi`).
  - **Multivariate walk-forward HMM** precomputed.
  - **Results**: dynamic BL Sharpe 0.68, MaxDD -14.45%.

- **v1.0** (2026-09-19): Initial release.
  - Complete pipeline: Markowitz -> GMP -> Reverse Optimization -> BL -> dynamic BL.
  - **Walk-forward HMM** (multivariate: equity, bonds, gold).
  - **CPI vintage (ALFRED)** for point-in-time inflation views.
  - **Lag-1 GMP weights** to remove look-ahead.
  - **Transaction costs** included (5-15 bps per asset class).
  - **Sharpe with time-varying risk-free rate**.
  - Out-of-sample backtest 2013-2026; the current suite has 65 passing tests
    and 1 skipped test.

---

*Last updated: 2026-10-07*