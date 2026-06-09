"""
Feature engineering for limit order book data.
Phase 3 implementation for quantitative market microstructure research.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features from processed order book data."""

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol

    def load_processed_data(self, file_path: str) -> pd.DataFrame:
        """Load processed data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} processed data points")
            return df
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return pd.DataFrame()

    def compute_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute spread feature: Spread = Ask - Bid."""
        if 'spread' in df.columns:
            logger.info("Spread already exists, skipping")
            return df
        if 'best_ask' in df.columns and 'best_bid' in df.columns:
            df['spread'] = df['best_ask'] - df['best_bid']
            logger.info("Computed spread feature")
        return df

    def compute_relative_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute relative spread feature: Spread / Midprice."""
        if 'relative_spread' in df.columns:
            return df
        if 'spread' in df.columns and 'midprice' in df.columns:
            df['relative_spread'] = df['spread'] / df['midprice']
            logger.info("Computed relative spread feature")
        return df

    def compute_depth_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute depth imbalance feature: (BidVolume - AskVolume) / (BidVolume + AskVolume)."""
        if 'depth_imbalance' in df.columns:
            logger.info("Depth imbalance already exists, skipping")
            return df
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            total_volume = df['bid_volume'] + df['ask_volume']
            df['depth_imbalance'] = (df['bid_volume'] - df['ask_volume']) / total_volume
            logger.info("Computed depth imbalance feature")
        return df

    def compute_order_flow_imbalance(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Compute order flow imbalance as rolling average of depth imbalance changes."""
        if 'order_flow_imbalance' in df.columns:
            return df
        if 'depth_imbalance' in df.columns:
            df['order_flow_imbalance'] = df['depth_imbalance'].rolling(window=window).mean()
            df['order_flow_imbalance'] = df['order_flow_imbalance'].fillna(0)
            logger.info(f"Computed order flow imbalance feature (window={window})")
        return df

    def compute_volatility(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Compute rolling realized volatility."""
        if 'volatility' in df.columns:
            return df
        if 'midprice' in df.columns:
            returns = np.log(df['midprice'] / df['midprice'].shift(1))
            df['volatility'] = returns.rolling(window=window).std() * np.sqrt(window)
            df['volatility'] = df['volatility'].fillna(0)
            logger.info(f"Computed volatility feature (window={window})")
        return df

    def compute_liquidity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute liquidity metrics: dollar depth, volume, trade count."""
        has_bid = 'bid_volume' in df.columns and 'best_bid' in df.columns
        has_ask = 'ask_volume' in df.columns and 'best_ask' in df.columns

        if 'dollar_depth' not in df.columns and has_bid and has_ask:
            df['dollar_depth'] = df['bid_volume'] * df['best_bid'] + df['ask_volume'] * df['best_ask']
            logger.info("Computed dollar depth")

        if 'volume' not in df.columns:
            if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
                df['volume'] = df['bid_volume'] + df['ask_volume']
            else:
                df['volume'] = 1.0
            logger.info("Computed volume metric")

        if 'trade_count' not in df.columns:
            df['trade_count'] = 1

        return df

    def engineer_all_features(self, input_file: str, output_file: str | None = None) -> pd.DataFrame:
        """Engineer all features from processed data."""
        logger.info("Starting feature engineering pipeline")

        df = self.load_processed_data(input_file)
        if df.empty:
            return df

        df = self.compute_spread(df)
        df = self.compute_relative_spread(df)
        df = self.compute_depth_imbalance(df)
        df = self.compute_order_flow_imbalance(df)
        df = self.compute_volatility(df)
        df = self.compute_liquidity_metrics(df)

        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Features saved to {output_file}")

        logger.info(f"Feature engineering complete. Final dataset has {len(df)} rows and {len(df.columns)} features")
        return df


def main():
    """Main entry point for feature engineering."""
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")

    args = parser.parse_args()

    engineer = FeatureEngineer()
    engineer.engineer_all_features(input_file=args.input, output_file=args.output)


if __name__ == "__main__":
    main()
