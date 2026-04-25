"""
Calibrated synthetic data generator matching Kraken BTC/USD statistical properties.

The real Kraken data shows:
- Bimodal imbalance (53% near ±1, from one-sided book)
- AR(1) ≈ 0.86 (0.92 when extreme, 0.58 when moderate)
- Returns with kurtosis ≈ 28, skew ≈ 2.3
- Spread mean ≈ 0.48, right-skewed

We model this as a two-state process:
  State 0: balanced book — both sides have volume, imb near 0
  State 1: one-sided book — one side dominates, imb near ±1
"""

import numpy as np
import pandas as pd
from typing import Optional


class CalibratedDataGenerator:
    """Generate synthetic LOB data calibrated to real Kraken BTC/USD."""

    def __init__(
        self,
        n_rows: int = 10000,
        interval_ms: float = 140.0,
        seed: int = 42,
        imb_state_ar1: float = 0.92,
        imb_balanced_ar1: float = 0.58,
        imb_std_balanced: float = 0.55,
        return_std: float = 4.5e-5,
        corr_imb_return_10s: float = 0.28,
        spread_rate: float = 0.48,
        p_stay_one_sided: float = 0.95,
        p_stay_balanced: float = 0.945,
        prob_one_sided: float = 0.53,
    ):
        self.n_rows = n_rows
        self.interval_ms = interval_ms
        self.imb_state_ar1 = imb_state_ar1
        self.imb_balanced_ar1 = imb_balanced_ar1
        self.imb_std_balanced = imb_std_balanced
        self.return_std = return_std
        self.corr_imb_return_10s = corr_imb_return_10s
        self.spread_rate = spread_rate
        self.p_stay_one_sided = p_stay_one_sided
        self.p_stay_balanced = p_stay_balanced
        self.prob_one_sided = prob_one_sided
        self.rng = np.random.default_rng(seed)

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        n = self.n_rows
        base_ts = int(pd.Timestamp("2026-06-10 22:00:00").timestamp() * 1000)
        timestamps = np.arange(n, dtype=np.int64) * int(self.interval_ms) + base_ts

        # 1. Two-state Markov chain
        state = np.zeros(n, dtype=np.int8)
        state[0] = 1 if self.rng.random() < self.prob_one_sided else 0
        for t in range(1, n):
            if state[t - 1] == 1:
                state[t] = 1 if self.rng.random() < self.p_stay_one_sided else 0
            else:
                state[t] = 1 if self.rng.random() >= self.p_stay_balanced else 0

        # 2. Imbalance
        imb = np.zeros(n)

        # State 1: near ±1, sign-persistent
        # Sign Markov chain: stays in +1 or -1 for long runs
        sign = np.ones(n)
        sign[0] = 1 if self.rng.random() < 0.5 else -1
        for t in range(1, n):
            sign[t] = sign[t - 1] if self.rng.random() < 0.97 else -sign[t - 1]
        # Noise around ±1 (small — stays near extremes)
        noise = self.rng.exponential(0.02, n)
        imb_state = sign * (1 - noise)
        imb_state = np.clip(imb_state, -1, 1)

        # State 0: near 0, moderate persistence
        phi0 = self.imb_balanced_ar1
        raw0 = np.empty(n)
        raw0[0] = self.rng.normal(0, self.imb_std_balanced)
        for t in range(1, n):
            raw0[t] = phi0 * raw0[t - 1] + self.rng.normal(0, self.imb_std_balanced * np.sqrt(1 - phi0 ** 2))
        raw0 = np.clip(raw0, -0.80, 0.80)

        # Combine states
        imb = np.where(state == 1, imb_state, raw0)

        # 3. Returns
        beta = self.corr_imb_return_10s * self.return_std / max(imb.std(), 1e-10)
        noise_var = max(self.return_std ** 2 - (beta * imb.std()) ** 2, 1e-12)
        epsilon = self.rng.standard_t(df=4, size=n) * np.sqrt(noise_var / 2.0)
        r_1s = beta * imb + epsilon

        # 4. Midprice
        midprice = np.empty(n)
        midprice[0] = 62000.0
        for t in range(1, n):
            midprice[t] = midprice[t - 1] * np.exp(r_1s[t])

        # 5. Spread (exponential, right-skewed)
        spread = self.rng.exponential(self.spread_rate, n)
        spread = np.maximum(spread, 0.01)

        # 6. Volumes — directly encode imbalance from generated values
        # Imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) = imb_generated
        # So: bid_vol = base_vol * (1 + imb), ask_vol = base_vol * (1 - imb)
        base_vol = self.rng.exponential(5, n)
        bid_vol = base_vol * (1 + imb)
        ask_vol = base_vol * (1 - imb)
        bid_vol = np.maximum(bid_vol, 0.001)
        ask_vol = np.maximum(ask_vol, 0.001)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "best_bid": midprice - spread / 2,
            "best_ask": midprice + spread / 2,
            "bid_volume": bid_vol,
            "ask_volume": ask_vol,
            "_imb_generated": imb,
        })

        self._check(df)
        return df

    def _check(self, df: pd.DataFrame) -> None:
        imb_from_vol = (df["bid_volume"] - df["ask_volume"]) / (df["bid_volume"] + df["ask_volume"])
        imb = df["_imb_generated"]
        extreme = np.abs(imb) > 0.95
        print(f"=== Calibration Check (n={len(df)}) ===")
        print(f"  AR(1):           {imb.corr(imb.shift(1)):.3f} (target: 0.86)")
        print(f"  Imb % extreme:   {extreme.mean()*100:.1f}% (target: 53%)")
        print(f"  Imb std:         {imb.std():.3f} (target: 0.84)")
        print(f"  Imb mean:        {imb.mean():.3f} (target: 0.21)")
        print(f"  AR(1) extreme:   {imb[extreme].corr(imb.shift(1)[extreme]):.3f} (target: 0.92)")
        print(f"  AR(1) moderate:  {imb[~extreme].corr(imb.shift(1)[~extreme]):.3f} (target: 0.58)")
        print(f"  Vol-match err:   {np.mean(np.abs(imb - imb_from_vol)):.6f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--output", default="data/raw/calibrated_synthetic.csv")
    args = parser.parse_args()
    gen = CalibratedDataGenerator(n_rows=args.rows)
    df = gen.generate()
    cols = ["timestamp", "best_bid", "best_ask", "bid_volume", "ask_volume"]
    df[cols].to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
