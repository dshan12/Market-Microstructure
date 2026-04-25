"""
Data processing pipeline for limit order book data.
Phase 2 implementation for quantitative market microstructure research.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OrderBookDataProcessor:
    """Process raw order book data into research-ready format."""

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load raw order book data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, forward-fill gaps, drop bad rows."""
        if df.empty:
            return df
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
        cols = ['best_bid', 'best_ask', 'bid_volume', 'ask_volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'best_bid' in df.columns and 'best_ask' in df.columns:
            df = df.dropna(subset=['best_bid', 'best_ask'])
        logger.info(f"After cleaning: {len(df)} rows")
        return df

    def compute_midprice(self, df: pd.DataFrame) -> pd.DataFrame:
        df['midprice'] = (df['best_bid'] + df['best_ask']) / 2
        logger.info("Computed midprice")
        return df

    def compute_future_returns(self, df: pd.DataFrame,
                               horizons: list = None) -> pd.DataFrame:
        """Compute future log returns using searchsorted for time-based lookup.

        For each row, finds the price at (timestamp + horizon) via binary
        search on the sorted millisecond timestamps.  Correct for any
        sampling frequency — no row-shifting assumptions.
        """
        if horizons is None:
            horizons = [1, 5, 10, 30, 60, 300]
        if df.empty or 'midprice' not in df.columns or 'timestamp' not in df.columns:
            return df

        df = df.sort_values('timestamp').reset_index(drop=True)
        ts = df['timestamp'].values        # milliseconds since epoch
        mid = df['midprice'].values

        for h in horizons:
            target = ts + h * 1000
            idx = np.searchsorted(ts, target, side='left')
            idx = np.clip(idx, 0, len(mid) - 1)
            future_mid = mid[idx]
            df[f'return_{h}s'] = np.log(future_mid / mid)

        logger.info(f"Computed returns for horizons: {horizons}")
        return df

    def process(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        logger.info("Starting data processing pipeline")
        df = self.load_data(input_file)
        if df.empty:
            return df
        df = self.clean(df)
        df = self.compute_midprice(df)
        df = self.compute_future_returns(df)
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Processed data saved to {output_file}")
        logger.info(f"Processing complete. {len(df)} rows")
        return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Order Book Data Processing Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()
    processor = OrderBookDataProcessor()
    processor.process(input_file=args.input, output_file=args.output)


if __name__ == "__main__":
    main()
