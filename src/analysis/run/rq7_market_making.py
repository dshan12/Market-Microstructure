"""
RQ7: Profitable market making strategies using order book imbalance.

Builds on RQ4 by testing advanced strategies designed to turn a profit:
  1. SkewGridSearch — find the optimal skew strength
  2. OneSidedMM — skip trades on the side that imbalance predicts will lose
  3. VelocityCautiousMM — reduce exposure when velocity predicts reversal
  4. CombinedSmartMM — best elements from each
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from .market_making import BasicMarketMaker, AdaptiveMarketMaker, MarketMakingSimulator

logger = logging.getLogger(__name__)


class OneSidedMarketMaker(BasicMarketMaker):
    """Only trades on the side favored by imbalance.

    Key insight from RQ4: the basic MM loses because it always provides
    two-sided quotes and gets picked off on the wrong side. When imbalance
    is high, we simply refuse to trade on the losing side — we withdraw
    that quote and only provide liquidity where we have an edge.

    - imb > threshold: only SELL (price likely rising, sell into demand)
    - imb < -threshold: only BUY (price likely falling, buy at discount)
    - |imb| <= threshold: trade both sides normally
    """

    def __init__(self, threshold: float = 0.3, position_limit: int = 50,
                 taker_fee: float = 0.0):
        super().__init__(position_limit=position_limit, taker_fee=taker_fee)
        self.threshold = threshold
        self.name = "OneSided"

    def compute_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        trades = []
        inventory = 0.0
        cash = 0.0

        df = df.copy()
        if 'depth_imbalance' not in df.columns:
            df['depth_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        df['spread'] = df['best_ask'] - df['best_bid']

        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_ret = df['return_1s'].iloc[i + 1]
            bid, ask = self.quote(row)

            if pd.isna(next_ret):
                continue

            mid = row['midprice']
            imb = row.get('depth_imbalance', 0)

            # → simplify: always quote both sides normally for price reference
            # but skip trades on the side we want to avoid
            should_trade = True
            direction = 0
            trade_price = np.nan

            if next_ret > 0 and inventory > -self.position_limit:
                # Ask gets hit (MM sells)
                if imb > self.threshold:
                    # Price rising AND we have buy pressure → don't sell into it
                    should_trade = False
                else:
                    trade_price = ask
                    cash += trade_price
                    inventory -= 1
                    direction = -1
            elif next_ret < 0 and inventory < self.position_limit:
                # Bid gets hit (MM buys)
                if imb < -self.threshold:
                    # Price falling AND we have sell pressure → don't buy into it
                    should_trade = False
                else:
                    trade_price = bid
                    cash -= trade_price
                    inventory += 1
                    direction = 1
            elif next_ret == 0:
                should_trade = False

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
                'spread': row.get('spread', ask - bid),
                'depth_imbalance': imb,
                'should_trade': should_trade,
            })

        result = pd.DataFrame(trades)
        if len(result) > 0:
            result['cumulative_pnl'] = result['pnl'].cumsum()
        return result


class VelocityCautiousMarketMaker(BasicMarketMaker):
    """Uses RQ6 finding: when imbalance is high AND velocity is positive,
    the signal is about to reverse. Skip trades during reversal risk.

    Also applies directional skew based on imbalance.
    """

    def __init__(self, skew_strength: float = 3.0, vol_mult: float = 2.0,
                 position_limit: int = 50, taker_fee: float = 0.0):
        super().__init__(position_limit=position_limit, taker_fee=taker_fee)
        self.skew_strength = skew_strength
        self.vol_mult = vol_mult
        self.name = "VelCautious"

    def compute_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        trades = []
        inventory = 0.0
        cash = 0.0

        df = df.copy()
        if 'depth_imbalance' not in df.columns:
            df['depth_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        df['imbalance_velocity'] = df['depth_imbalance'].diff(5) / 5
        df['spread'] = df['best_ask'] - df['best_bid']

        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_ret = df['return_1s'].iloc[i + 1]
            bid, ask = self.quote(row)

            if pd.isna(next_ret):
                continue

            mid = row['midprice']
            imb = row.get('depth_imbalance', 0)
            vel = row.get('imbalance_velocity', 0)
            reversal_risk = abs(imb) * max(0, imb * vel)

            should_trade = True
            direction = 0
            trade_price = np.nan

            # Skip if reversal risk is high (velocity same sign as high imbalance)
            if reversal_risk > 0.5 and np.abs(imb) > 0.5:
                should_trade = False

            if next_ret > 0 and inventory > -self.position_limit:
                if should_trade:
                    trade_price = ask
                    cash += trade_price
                    inventory -= 1
                    direction = -1
            elif next_ret < 0 and inventory < self.position_limit:
                if should_trade:
                    trade_price = bid
                    cash -= trade_price
                    inventory += 1
                    direction = 1
            elif next_ret == 0:
                should_trade = False

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
                'spread': row.get('spread', ask - bid),
                'depth_imbalance': imb,
                'imbalance_velocity': vel,
                'should_trade': should_trade,
                'reversal_risk': reversal_risk,
            })

        result = pd.DataFrame(trades)
        if len(result) > 0:
            result['cumulative_pnl'] = result['pnl'].cumsum()
        return result


class CombinedSmartMarketMaker(BasicMarketMaker):
    """Best elements combined:
    1. One-sided: skip losing-side trades when |imb| > threshold
    2. Velocity caution: skip trades during reversal risk
    3. Adaptive skew for remaining trades
    """

    def __init__(self, skew_strength: float = 3.0, vol_mult: float = 2.0,
                 one_sided_threshold: float = 0.3,
                 reversal_threshold: float = 0.5,
                 position_limit: int = 50, taker_fee: float = 0.0):
        super().__init__(position_limit=position_limit, taker_fee=taker_fee)
        self.skew_strength = skew_strength
        self.vol_mult = vol_mult
        self.one_sided_threshold = one_sided_threshold
        self.reversal_threshold = reversal_threshold
        self.name = "CombinedSmart"

    def compute_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        trades = []
        inventory = 0.0
        cash = 0.0

        df = df.copy()
        if 'depth_imbalance' not in df.columns:
            df['depth_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        df['imbalance_velocity'] = df['depth_imbalance'].diff(5) / 5
        df['volatility'] = df['return_1s'].rolling(20).std()
        df['spread'] = df['best_ask'] - df['best_bid']

        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_ret = df['return_1s'].iloc[i + 1]
            bid, ask = self.quote(row)

            if pd.isna(next_ret):
                continue

            mid = row['midprice']
            imb = row.get('depth_imbalance', 0)
            vel = row.get('imbalance_velocity', 0)
            reversal_risk = abs(imb) * max(0, imb * vel)

            should_trade = True
            direction = 0
            trade_price = np.nan

            # Skip: one-sided (trading against imbalance direction)
            if next_ret > 0 and imb > self.one_sided_threshold:
                should_trade = False
            elif next_ret < 0 and imb < -self.one_sided_threshold:
                should_trade = False

            # Skip: reversal risk
            if should_trade and reversal_risk > self.reversal_threshold:
                should_trade = False

            if next_ret > 0 and inventory > -self.position_limit:
                if should_trade:
                    trade_price = ask
                    cash += trade_price
                    inventory -= 1
                    direction = -1
            elif next_ret < 0 and inventory < self.position_limit:
                if should_trade:
                    trade_price = bid
                    cash -= trade_price
                    inventory += 1
                    direction = 1
            elif next_ret == 0:
                should_trade = False

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
                'spread': row.get('spread', ask - bid),
                'depth_imbalance': imb,
                'imbalance_velocity': vel,
                'should_trade': should_trade,
                'reversal_risk': reversal_risk,
            })

        result = pd.DataFrame(trades)
        if len(result) > 0:
            result['cumulative_pnl'] = result['pnl'].cumsum()
        return result


def grid_search_skew(df: pd.DataFrame, skew_values: List[float] = None,
                     vol_mult: float = 2.0) -> pd.DataFrame:
    """Grid search over skew_strength to find optimal parameter."""
    if skew_values is None:
        skew_values = [0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

    results = []
    for skew in skew_values:
        mm = AdaptiveMarketMaker(skew_strength=skew, vol_mult=vol_mult)
        mm.name = f"skew={skew}"
        trades = mm.compute_pnl(df)
        stats = MarketMakingSimulator._compute_stats(trades, mm.name)
        results.append(stats | {'skew': skew})

    return pd.DataFrame(results)


def grid_search_one_sided(df: pd.DataFrame, thresholds: List[float] = None) -> pd.DataFrame:
    """Grid search over OneSided threshold."""
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    results = []
    for thresh in thresholds:
        mm = OneSidedMarketMaker(threshold=thresh)
        mm.name = f"onesided_th={thresh}"
        trades = mm.compute_pnl(df)
        stats = MarketMakingSimulator._compute_stats(trades, mm.name)
        results.append(stats | {'threshold': thresh})

    return pd.DataFrame(results)


def run_rq7(input_file: str, output_prefix: str = "results/rq7") -> Dict:
    """Run all RQ7 strategy comparisons."""
    df = pd.read_csv(input_file)
    df['spread'] = df['best_ask'] - df['best_bid']
    df['depth_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])

    sim = MarketMakingSimulator(df)

    strategies = [
        BasicMarketMaker(),
        AdaptiveMarketMaker(skew_strength=3.0, vol_mult=2.0),
        OneSidedMarketMaker(threshold=0.3),
        VelocityCautiousMarketMaker(skew_strength=3.0),
        CombinedSmartMarketMaker(one_sided_threshold=0.3, reversal_threshold=0.5),
    ]

    results = sim.run(strategies)

    logger.info("Running skew grid search...")
    skew_grid = grid_search_skew(df)

    logger.info("Running one-sided threshold grid search...")
    onesided_grid = grid_search_one_sided(df)

    serializable = {}
    for name, r in results.items():
        serializable[name] = {'stats': r['stats']}
    serializable['skew_grid'] = skew_grid.to_dict('records')
    serializable['onesided_grid'] = onesided_grid.to_dict('records')

    import json, os
    os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)
    with open(f"{output_prefix}.json", 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"RQ7 — Market Making Strategy Comparison")
    print(f"{'='*60}")
    sim.compare()
    print(f"\nGrid Search — Skew (top 3 by PnL):")
    best_skew = skew_grid.sort_values('total_pnl', ascending=False).head(3)
    for _, row in best_skew.iterrows():
        print(f"  skew={row['skew']:5.1f}: PnL={row['total_pnl']:+8.2f}, "
              f"Sharpe={row['sharpe_ratio']:+.3f}, Trades={row['trade_count']}")
    print(f"\nGrid Search — OneSided Threshold (top 3 by PnL):")
    best_ones = onesided_grid.sort_values('total_pnl', ascending=False).head(3)
    for _, row in best_ones.iterrows():
        print(f"  threshold={row['threshold']:.1f}: PnL={row['total_pnl']:+8.2f}, "
              f"Sharpe={row['sharpe_ratio']:+.3f}, Trades={row['trade_count']}, "
              f"SpreadCap={row['spread_capture_pct']:.0f}%")

    return serializable


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RQ7: Profitable Market Making Strategies")
    parser.add_argument("--input", default="data/processed/kraken_final.csv")
    parser.add_argument("--output", default="results/rq7")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_rq7(args.input, args.output)


if __name__ == "__main__":
    main()
