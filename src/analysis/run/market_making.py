"""
Market making simulator for Phase 8 / RQ4.
Evaluates whether order book imbalance signals translate into
real economic value for a liquidity provider.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class BasicMarketMaker:
    """Naive market maker: symmetric quotes around midprice at the observed spread.

    Every step, a market order arrives and hits one side:
      - positive next return → buyer aggressor → ask gets hit (MM sells 1 unit)
      - negative next return → seller aggressor → bid gets hit (MM buys 1 unit)

    PnL tracks cash + inventory × midprice (mark-to-market).
    Trade size is 1 unit (abstract contract); PnL in quote currency units.
    """

    def __init__(self, position_limit: int = 50, taker_fee: float = 0.0):
        self.position_limit = position_limit
        self.taker_fee = taker_fee
        self.name = "Basic"

    def quote(self, row: pd.Series) -> Tuple[float, float]:
        spread = row['spread']
        mid = row['midprice']
        return mid - spread / 2, mid + spread / 2

    def compute_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        trades = []
        inventory = 0.0
        cash = 0.0

        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_ret = df['return_1s'].iloc[i + 1]
            bid, ask = self.quote(row)

            if pd.isna(next_ret):
                continue

            mid = row['midprice']

            if next_ret > 0 and inventory > -self.position_limit:
                trade_price = ask * (1 - self.taker_fee)
                cash += trade_price
                inventory -= 1
                direction = -1
            elif next_ret < 0 and inventory < self.position_limit:
                trade_price = bid * (1 + self.taker_fee)
                cash -= trade_price
                inventory += 1
                direction = 1
            else:
                trade_price = np.nan
                direction = 0

            mtm = cash + inventory * mid
            trades.append({
                'step': i,
                'midprice': mid,
                'bid': bid,
                'ask': ask,
                'trade_price': trade_price,
                'trade_sign': direction,
                'inventory': inventory,
                'cash': cash,
                'mtm': mtm,
                'pnl': mtm - trades[-1]['mtm'] if trades else 0.0,
                'spread': row['spread'],
                'depth_imbalance': row.get('depth_imbalance', 0),
                'volatility': row.get('volatility', 0),
            })

        result = pd.DataFrame(trades)
        if len(result) > 0:
            result['cumulative_pnl'] = result['pnl'].cumsum()
        return result


class AdaptiveMarketMaker(BasicMarketMaker):
    """Adaptive market maker that skews quotes based on order book signals.

    Strategy:
      - Skew: shift both quotes toward the direction of imbalance,
             making it cheaper to buy (add to inventory) when buy pressure detected,
             and cheaper to sell (reduce inventory) when sell pressure detected.
             This is the opposite of the naive approach — we lean with the flow.
      - Widen spreads during high volatility for protection.
      - Position limits to control risk.
    """

    def __init__(self, skew_strength: float = 5.0, vol_mult: float = 3.0,
                 position_limit: int = 50, taker_fee: float = 0.0):
        super().__init__(position_limit=position_limit, taker_fee=taker_fee)
        self.skew_strength = skew_strength
        self.vol_mult = vol_mult
        self.name = "Adaptive"

    def quote(self, row: pd.Series) -> Tuple[float, float]:
        base_spread = row['spread']
        imbalance = row.get('depth_imbalance', 0)
        volatility = row.get('volatility', 0)
        mid = row['midprice']

        vol_scale = max(1.0, self.vol_mult * volatility / max(volatility, 1e-12))
        spread = base_spread * min(vol_scale, 5.0)

        skew = imbalance * self.skew_strength * base_spread

        ask = mid + spread / 2 + skew
        bid = mid - spread / 2 + skew

        if bid >= ask:
            bid = mid - spread / 2
            ask = mid + spread / 2

        return bid, ask


class MarketMakingSimulator:
    """Run market making simulations comparing Basic vs Adaptive strategies."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.dropna(subset=['midprice', 'spread', 'return_1s']).reset_index(drop=True)

    def run(self, strategies: Optional[list] = None) -> Dict:
        if strategies is None:
            strategies = [BasicMarketMaker(), AdaptiveMarketMaker()]

        results = {}
        for mm in strategies:
            logger.info(f"Running {mm.name} market maker...")
            trades = mm.compute_pnl(self.df)
            stats = self._compute_stats(trades, mm.name)
            results[mm.name] = {
                'trades': trades,
                'stats': stats
            }
            logger.info(f"  {mm.name} MM: PnL={stats['total_pnl']:.2f}, "
                        f"Sharpe={stats['sharpe_ratio']:.3f}, "
                        f"MaxDD={stats['max_drawdown_pct']:.2%}, "
                        f"SpreadCapture={stats['spread_capture_pct']:.1f}%")

        self.results = results
        return results

    @staticmethod
    def _compute_stats(trades: pd.DataFrame, name: str) -> Dict:
        if trades.empty or len(trades) < 10:
            return {}

        pnl_series = trades['pnl'].fillna(0)
        cum_pnl = pnl_series.cumsum()
        peak = cum_pnl.expanding().max()
        drawdown = cum_pnl - peak

        total_pnl = float(cum_pnl.iloc[-1]) if len(cum_pnl) > 0 else 0.0
        sharpe = float(pnl_series.mean() / pnl_series.std() * np.sqrt(252 * 6 * 60 * 10)
                       ) if pnl_series.std() > 0 else 0.0
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
        max_dd_pct = float(drawdown.min() / (peak.max() + 1e-12)) if len(drawdown) > 0 else 0.0
        max_dd_pct = max(min(max_dd_pct, 0.0), -1.0)
        inventory_vol = float(trades['inventory'].std())
        trade_count = int((trades['trade_sign'] != 0).sum())
        avg_spread = float(trades['spread'].mean())

        trades_only = trades[trades['trade_sign'] != 0]
        gross_spread_pnl = float((trades_only['ask'] - trades_only['bid']).sum()) if len(trades_only) > 0 else 0.0
        inventory_pnl = total_pnl - gross_spread_pnl

        return {
            'total_pnl': round(total_pnl, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_dd, 2),
            'max_drawdown_pct': round(max_dd_pct, 4),
            'gross_spread_pnl': round(gross_spread_pnl, 2),
            'inventory_pnl': round(inventory_pnl, 2),
            'inventory_volatility': round(inventory_vol, 4),
            'trade_count': trade_count,
            'avg_spread': round(avg_spread, 6),
            'spread_capture_pct': round(trade_count / len(trades) * 100, 1),
            'n_steps': len(trades)
        }

    def compare(self) -> pd.DataFrame:
        """Return a comparison table of strategy stats."""
        rows = []
        for name, result in self.results.items():
            s = result['stats']
            rows.append({
                'Strategy': name,
                'Total PnL': f"{s.get('total_pnl', 0):+.2f}",
                'Sharpe': f"{s.get('sharpe_ratio', 0):.3f}",
                'Max DD': f"{s.get('max_drawdown', 0):.2%}",
                'Inv Vol': f"{s.get('inventory_volatility', 0):.4f}",
                'Trades': s.get('trade_count', 0),
                'Spread Capture': f"{s.get('spread_capture_pct', 0):.1f}%",
                'Steps': s.get('n_steps', 0),
            })
        return pd.DataFrame(rows)


def run_simulation(input_file: str = "data/features/btcusdt_features.csv") -> Dict:
    """Convenience function: load data and run MM simulation."""
    df = pd.read_csv(input_file)
    sim = MarketMakingSimulator(df)
    strategies = [
        BasicMarketMaker(),
        AdaptiveMarketMaker(skew_strength=0.5, vol_mult=2.0),
    ]
    results = sim.run(strategies)
    sim.compare()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Market Making Simulation")
    parser.add_argument("--input", default="data/features/btcusdt_features.csv")
    parser.add_argument("--output", default="results/market_making.json")
    args = parser.parse_args()

    import json
    results = run_simulation(args.input)

    serializable = {}
    for name, r in results.items():
        serializable[name] = {
            'stats': r['stats'],
        }

    with open(args.output, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Market making results saved to {args.output}")


if __name__ == "__main__":
    main()
