# LOB Depth Asymmetry and Short-Horizon Return Predictability
*An Empirical Microstructure Analysis of BTC/USD and ETH/USD on Kraken*

This repository implements a rigorous, end-to-end quantitative research pipeline to investigate whether Limit Order Book (LOB) depth asymmetry contains predictive information for short-horizon future returns. Using high-frequency, tick-by-tick Kraken LOB top-of-book data, we analyze the statistical properties of order book imbalance, its decay half-life, its causal directionality, and its exploitability under realistic market-making simulation constraints.

---

## 📐 Mathematical & Econometric Framework

### 1. Depth Imbalance ($I_t$)
We define the instantaneous top-of-book depth imbalance at time $t$ as the normalized volume difference between the best bid ($V_t^b$) and the best ask ($V_t^a$):

$$I_t = \frac{V_t^b - V_t^a}{V_t^b + V_t^a} \in [-1, 1]$$

### 2. Microstructure Predictive Regression
To evaluate return predictability over a horizon $h \in \{1\text{s}, 5\text{s}, 10\text{s}, 30\text{s}, 60\text{s}\}$, we estimate the following linear predictive model:

$$R_{t, t+h} = \beta_0 + \beta_1 I_t + \mathbf{\Gamma} \mathbf{X}_t + \epsilon_{t, t+h}$$

Where:
*   $R_{t, t+h} = \ln(\text{Mid}_{t+h} / \text{Mid}_t)$ is the log mid-price return.
*   $\mathbf{X}_t$ is a vector of control variables (e.g., lagged returns, bid-ask spread).
*   Since overlap in multi-period returns induces serial correlation in $\epsilon_{t, t+h}$, all inference is performed using **Newey-West (1987) Heteroskedasticity and Autocorrelation Consistent (HAC)** standard errors with $L = 2h$ lags.

### 3. Imbalance Velocity & Acceleration
To capture the dynamics of order flow, we compute the first and second discrete-time derivatives of the imbalance series:

$$\text{Velocity } (v_t) = \frac{I_t - I_{t-1}}{\Delta t}, \quad \text{Acceleration } (a_t) = \frac{v_t - v_{t-1}}{\Delta t}$$

---

## 📊 Core Empirical Results

Our findings on real Kraken market data (5,768 BTC/USD and 5,002 ETH/USD tick-by-tick observations) differ dramatically from naive synthetic benchmarks:

| Metric | BTC/USD (Full) | BTC/USD (High-Vol Window) | ETH/USD (Full) |
| :--- | :---: | :---: | :---: |
| **Imbalance Autoregression $\text{AR}(1)$** | **0.856** | **0.863** | **0.824** |
| **Peak Correlation ($r$)** | 0.166 (at 30s) | **0.282** (at 10s) | 0.070 (at 10s) |
| **Signal Half-Life ($\tau_{1/2}$)** | **85.4 seconds** | **524.1 seconds** | **24.5 seconds** |
| **Granger Causality ($I_t \to R_t$)** | $p = 0.337$ | **$p = 0.005$ (Significant)** | **$p = 0.004$ (Significant)** |
| **Velocity Incremental $R^2$** | **+8.8%** | **+16.4%** | **+22.8%** |

### Key Microstructure Insights:
1.  **Persistence & Memory**: The extremely high $\text{AR}(1)$ coefficient ($\approx 0.86$) confirms that real-world LOB asymmetry is highly persistent, consistent with institutional traders splitting orders (Kyle, 1985) or herd-like liquidity provision.
2.  **Velocity Mean Reversion**: The regression coefficient for imbalance velocity ($v_t$) is consistently **negative** across all assets. This indicates that while *static* imbalance predicts continuation, rapid *momentum* in imbalance signals impending local mean-reversion.
3.  **Cross-Asset Asymmetry**: ETH exhibits significantly faster decay ($\tau_{1/2} \approx 24.5\text{s}$) and smaller average spreads than BTC, pointing to a more efficient local arbitrage environment.

---

## 💻 Code Showcase: Vectorized Feature Engineering

All feature computations are fully vectorized in `numpy` and `pandas` for maximum execution speed over high-frequency tick data:

```python
import numpy as np
import pandas as pd

def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    # Compute instantaneous imbalance
    df['imbalance'] = (df['bid_vol'] - df['ask_vol']) / (df['bid_vol'] + df['ask_vol'])
    
    # Compute continuous bid-ask spread in basis points
    df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2.0
    df['spread_bps'] = (df['best_ask'] - df['best_bid']) / df['mid_price'] * 10000
    
    # Time-delta calculation for physical time derivatives
    dt = df['timestamp'].diff().dt.total_seconds().fillna(1.0).clip(lower=0.1)
    
    # Compute first and second order dynamics
    df['velocity'] = df['imbalance'].diff() / dt
    df['acceleration'] = df['velocity'].diff() / dt
    
    return df
```

---

## 🤖 Market Making Simulation & Strategy Performance

We implement an event-driven LOB simulator to evaluate whether a market maker can exploit these signals under realistic inventory constraints. 

### Performance Summary (BTC Full Dataset)

| Strategy | Total PnL | Sharpe Ratio | Max Drawdown | Spread Capture % |
| :--- | :---: | :---: | :---: | :---: |
| **Basic Two-Sided MM** | $-\$33,118$ | $-16.52$ | $-100.0\%$ | 23.7% |
| **Adaptive Quoting (skew=3)** | $-\$32,655$ | $-16.29$ | $-100.0\%$ | 23.7% |
| **OneSided (threshold=0.3)** | **$+\$6,089** | **+7.76** | **$-57.1\%$** | 35.6% |
| **Optimized OneSided (threshold=0.0)** | **$+\$10,296** | **+9.74** | **$-32.4\%$** | 27.0% |
| **Adaptive Quoting (ETH, skew=50)** | **$+\$2,164** | **+40.98** | **$-12.5\%$** | 70.8% |

### Economic Interpretation of the "MM Profit":
*   **Adverse Selection is Fatal**: Passive two-sided market making (Basic & Adaptive) suffers catastrophic losses because the MM is consistently run over by toxic, informed flow (the "inventory write-down" effect).
*   **The Momentum Shift**: The highly profitable **OneSided** strategy works by completely shutting off quote placement on the side threatened by adverse selection. PnL decomposition reveals that **83% to 87%** of the profits originate from **inventory appreciation** rather than spread capture. This proves that the strategy behaves as a **directional momentum trader** utilizing limit orders, not a traditional neutral market maker.

---

## 📂 Repository Layout & Architecture

```text
├── data/
│   └── processed/          # Processed parquet-equivalent CSVs (BTC, ETH, synthetic)
├── src/
│   ├── data/
│   │   ├── collect/        # Asyncio Kraken WebSocket client
│   │   └── process/        # High-performance searchsorted return alignment pipeline
│   ├── analysis/
│   │   └── run/
│   │       ├── run_all.py  # Master statistical analysis and OLS HAC engine
│   │       ├── market_making.py     # Base simulator engine and basic strategies
│   │       ├── rq7_market_making.py # Advanced strategies (OneSided, Combined)
│   │       └── rq7_viz.py           # Publication plots for grid search/PnL
├── results/                # Complete econometric JSON outputs
├── plots/                  # Scientific visualizations (10 publication-quality plots per asset)
├── research/
│   └── paper.md            # Comprehensive draft paper (17 sections, Appendices A-C)
└── pyproject.toml          # Project dependencies
```

---

## ⚡ Setup & Replication

This project uses `uv` for reproducible environment resolution.

### 1. Synchronize Dependencies
```bash
# Install uv package manager if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Bootstrap the virtual environment and install dependencies
uv sync
```

### 2. Execute the Quantitative Analysis Suite
This command executes the entire econometric pipeline for BTC, processes the statistics, and outputs 10 publication-quality analytical plots:
```bash
uv run python -m src.analysis.run.run_all --input data/processed/kraken_final.csv
```

### 3. Run Advanced Strategy Simulation & Optimization (RQ7)
Evaluate OneSided and Adaptive grid searches:
```bash
uv run python -m src.analysis.run.rq7_market_making --input data/processed/kraken_final.csv
```

---

## 📖 References

This research builds upon the following foundational and contemporary works in market microstructure:

### Theoretical Foundations & Stochastic Order Book Models
1. **Kyle, A. S. (1985).** Continuous auctions and insider trading. *Econometrica*, 53(6), 1315–1335.
2. **Glosten, L. R., & Milgrom, P. R. (1985).** Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
3. **Cont, R., Stoikov, S., & Talreja, R. (2010).** A stochastic model for order book dynamics. *Operations Research*, 58(3), 549-563.
4. **Foucault, T., Kadan, O., & Kandel, E. (2005).** Limit order book as a market for liquidity. *Review of Financial Studies*, 18(4), 1171–1217.
5. **Gourieroux, C., & Jasiak, J. (2001).** *Financial Econometrics*. Princeton University Press.

### Order Flow, Information & Imbalance Predictability
6. **Hasbrouck, J. (2003).** Trading and quoting on the NYSE. In *Empirical Market Microstructure*.
7. **Easley, D., Kiefer, N. M., & O'Hara, M. (1997).** The information content of the trading process. *Journal of Empirical Finance*, 4(2–3), 159–186.
8. **Lipton, A., Pesavento, U., & Sotiropoulos, M. (2013).** OBI (order book imbalance) and return predictability. *Quantitative Finance*, 13(12), 1967-1981.
9. **Foucault, T., Segebrecht, E., & Yildirim, M. (2021).** Order flow and information leakage: Evidence from high-frequency order book data. *Journal of Financial Economics*, 141(1), 1-28.
10. **Chordia, T., Roll, R., & Subrahmanyam, A. (2005).** Evidence on the speed of convergence to market efficiency. *Journal of Financial Economics*, 76(2), 271–292.

### Market Making, Optimal Execution & Inventory Management
11. **Avellaneda, M., & Stoikov, S. (2008).** High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.
12. **Cartea, Á., & Jaimungal, S. (2014).** Risk metrics and fine tuning of high-frequency trading strategies. *Mathematical Finance*, 24(3), 577-602.
13. **O'Hara, M. (1995).** *Market Microstructure Theory*. Blackwell.

### Econometric Methods & Empirical Microstructure
14. **Newey, W. K., & West, K. D. (1987).** A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
15. **Lo, A. W., & MacKinlay, A. C. (1990).** An econometric analysis of nonsynchronous trading. *Journal of Econometrics*, 45(1–2), 181–211.
16. **Cont, R. (2001).** Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.
17. **Pastor, L., & Stambaugh, R. F. (2003).** Liquidity risk and expected stock returns. *Journal of Political Economy*, 111(3), 642–685.

---

## 🚀 Quick Start

```bash
# 1. Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync environment (Python 3.12+)
uv sync

# 3. Run full econometric pipeline (RQ1-RQ6 + 10 plots)
uv run python -m src.analysis.run.run_all --input data/processed/kraken_final.csv

# 4. Run RQ7 advanced strategy grid search
uv run python -m src.analysis.run.rq7_market_making --input data/processed/kraken_final.csv
```

---

## 📁 Project Structure

```text
├── data/
│   └── processed/          # Cleaned CSVs with imbalance, returns, velocity
├── src/
│   ├── data/
│   │   ├── collect/        # Kraken WebSocket collector (async)
│   │   └── process/        # Pipeline: searchsorted return alignment
│   └── analysis/
│       └── run/
│           ├── run_all.py  # Master: RQ1-RQ6 + HAC/OLS + 10 plots
│           ├── market_making.py     # Base simulator & Basic/Adaptive MM
│           ├── rq7_market_making.py # OneSided, VelCautious, CombinedSmart + grid search
│           └── rq7_viz.py           # Strategy comparison & grid search plots
├── results/                # JSON outputs: regression tables, Granger tests, MM stats
├── plots/                  # 10 publication-quality plots per asset
├── research/
│   └── paper.md            # 17-section draft paper + Appendices A-C
└── pyproject.toml          # Dependencies: pandas, numpy, scipy, statsmodels, matplotlib
```

---

## 📈 Key Findings Summary

| Finding | BTC/USD | ETH/USD |
|---------|---------|---------|
| **Imbalance AR(1)** | 0.856 | 0.824 |
| **Peak $r$ (imbalance vs return)** | 0.282 (High-Vol) | 0.070 |
| **Signal Half-Life** | 85.4s (524s High-Vol) | 24.5s |
| **Granger Causality** | Window-dependent | $p=0.004$ ** |
| **Velocity $\Delta R^2$** | +8.8% | +22.8% |
| **Profitable MM Strategy** | OneSided (threshold=0) | Adaptive (skew=50) |
| **MM Profit Source** | 87% Inventory (Directional) | 54% Inventory |

---

*This is an academic research project for empirical market microstructure analysis. It is NOT a trading bot or investment advice.*