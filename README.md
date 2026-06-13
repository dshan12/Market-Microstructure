# LOB Depth Asymmetry and Short-Horizon Return Predictability

An empirical market microstructure analysis of BTC/USD and ETH/USD on Kraken.

This project investigates whether Limit Order Book (LOB) depth asymmetry contains predictive information for short-horizon future returns. We analyze 5,768 BTC/USD and 5,002 ETH/USD tick-by-tick observations from Kraken to study the statistical properties of order book imbalance, its decay half-life, causal directionality, and exploitability under market-making constraints.

## Mathematical Framework

### Depth Imbalance (I_t)

We define the instantaneous top-of-book depth imbalance as the normalized volume difference between best bid and best ask:

I_t = (V_t^b - V_t^a) / (V_t^b + V_t^a), range [-1, 1]

### Microstructure Predictive Regression

For horizon h in {1s, 5s, 10s, 30s, 60s} we estimate:

R_{t,t+h} = beta_0 + beta_1 * I_t + Gamma * X_t + epsilon_{t,t+h}

All inference uses Newey-West (1987) HAC standard errors with L = 2h lags due to overlapping return-induced serial correlation.

### Imbalance Velocity and Acceleration

To capture order flow dynamics:

Velocity (v_t) = (I_t - I_{t-1}) / Delta_t
Acceleration (a_t) = (v_t - v_{t-1}) / Delta_t

## Core Results

| Metric | BTC/USD | BTC/USD (High-Vol) | ETH/USD |
|--------|---------|-------------------|---------|
| AR(1) | 0.856 | 0.863 | 0.824 |
| Peak r | 0.166 (30s) | 0.282 (10s) | 0.070 (10s) |
| Half-life | 85.4s | 524.1s | 24.5s |
| Granger p | 0.337 | 0.005 | 0.004 |
| Velocity Delta R^2 | +8.8% | +16.4% | +22.8% |

### Key Observations

The high AR(1) coefficient (~0.86) confirms real-world LOB asymmetry is highly persistent, consistent with institutional order splitting (Kyle, 1985). The consistently negative velocity coefficient suggests rapid changes in imbalance signal impending mean-reversion. ETH shows faster signal decay and smaller spreads than BTC, suggesting a more efficient local arbitrage environment.

## Strategy Performance

We implemented an event-driven LOB simulator to evaluate market making strategies.

| Strategy | Total PnL | Sharpe | Max DD | Spread Capture |
|----------|-----------|--------|--------|----------------|
| Basic Two-Sided MM | -$33,118 | -16.52 | -100% | 23.7% |
| Adaptive (skew=3) | -$32,655 | -16.29 | -100% | 23.7% |
| OneSided (th=0.3) | +$6,089 | +7.76 | -57.1% | 35.6% |
| OneSided (th=0.0) | +$10,296 | +9.74 | -32.4% | 27.0% |
| Adaptive ETH (skew=50) | +$2,164 | +40.98 | -12.5% | 70.8% |

The profitable strategies turn out to be directional momentum strategies in disguise. PnL decomposition shows 83-87% of profits come from inventory appreciation, not spread capture.

## Repository Structure

```
data/processed/          Processed CSVs with imbalance and returns
src/data/collect         Kraken WebSocket client
src/data/process         searchsorted return alignment pipeline
src/analysis/run         RQ analysis, market making simulators, visualizations
results/                 Econometric output in JSON
plots/                   Publication figures
research/                Paper (LaTeX, PDF, BibTeX)
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python -m src.analysis.run.run_all --input data/processed/kraken_final.csv
uv run python -m src.analysis.run.rq7_market_making --input data/processed/kraken_final.csv
```

## References

Theoretical:
- Kyle, A. S. (1985). Continuous auctions and insider trading. Econometrica.
- Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices. JFE.
- Cont, R., Stoikov, S., & Talreja, R. (2010). A stochastic model for order book dynamics. Operations Research.
- Foucault, T., Kadan, O., & Kandel, E. (2005). Limit order book as a market for liquidity. RFS.

Order Flow and Imbalance:
- Hasbrouck, J. (2003). Empirical Market Microstructure. OUP.
- Lipton, A., Pesavento, U., & Sotiropoulos, M. (2013). OBI and return predictability. Quantitative Finance.
- Chordia, T., Roll, R., & Subrahmanyam, A. (2005). Speed of convergence to market efficiency. JFE.
- Easley, D., Kiefer, N. M., & O'Hara, M. (1997). Information content of the trading process. JEF.
- Foucault, T., Segebrecht, E., & Yildirim, M. (2021). Order flow and information leakage. JFE.

Market Making:
- Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. Quantitative Finance.
- Cartea, A., & Jaimungal, S. (2014). Risk metrics and fine tuning of HFT strategies. Mathematical Finance.
- O'Hara, M. (1995). Market Microstructure Theory. Blackwell.

Econometrics:
- Newey, W. K., & West, K. D. (1987). HAC covariance matrix. Econometrica.
- Lo, A. W., & MacKinlay, A. C. (1990). Nonsynchronous trading. Journal of Econometrics.
- Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance.
- Pastor, L., & Stambaugh, R. F. (2003). Liquidity risk and expected returns. JPE.

This is an academic research project. Not investment advice.
