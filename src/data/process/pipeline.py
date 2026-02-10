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
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries based on timestamp and symbol."""
        if df.empty:
            return df
        
        df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        logger.info(f"After duplicate removal: {len(df)} rows")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if df.empty:
            return df
        
        # Forward fill for best bid/ask prices
        numeric_cols = ['best_bid', 'best_ask', 'bid_volume', 'ask_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # Drop rows where essential data is missing
        df = df.dropna(subset=['best_bid', 'best_ask'])
        logger.info(f"After missing value handling: {len(df)} rows")
        return df
    
    def synchronize_timestamps(self, df: pd.DataFrame, target_frequency: str = "1s") -> pd.DataFrame:
        """Synchronize timestamps and resample data."""
        if df.empty:
            return df
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Select only numeric columns for resampling
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Resample to target frequency
        df_resampled = numeric_df.resample(target_frequency).mean()
        df_resampled.reset_index(inplace=True)
        
        logger.info(f"After timestamp synchronization: {len(df_resampled)} rows")
        return df_resampled
    
    def standardize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names, data types, and formats."""
        if df.empty:
            return df
        
        # Ensure standard column names
        column_mapping = {
            'best_bid': 'best_bid',
            'best_ask': 'best_ask',
            'bid_volume': 'bid_volume',
            'ask_volume': 'ask_volume',
            'symbol': 'symbol',
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure numeric types for key columns
        numeric_cols = ['best_bid', 'best_ask', 'bid_volume', 'ask_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure timestamp is numeric (milliseconds since epoch)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # Ensure symbol is string
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype(str)
        
        logger.info("Standardized data formats")
        return df
    
    def compute_midprice(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute midprice as primary price measure."""
        if df.empty:
            return df
        
        df['midprice'] = (df['best_bid'] + df['best_ask']) / 2
        logger.info("Computed midprice feature")
        return df
    
    def compute_future_returns(self, df: pd.DataFrame, horizons: list = [1, 5, 10, 30, 60, 300]) -> pd.DataFrame:
        """Compute future log returns for different horizons."""
        if df.empty:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        for horizon in horizons:
            shifted_price = df['midprice'].shift(-horizon)
            log_returns = (np.log(shifted_price) - np.log(df['midprice'])).shift(-horizon)
            df[f'return_{horizon}s'] = log_returns
        
        df.reset_index(inplace=True)
        logger.info(f"Computed future log returns for horizons: {horizons}")
        return df
    
    def process(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Complete data processing pipeline."""
        logger.info("Starting data processing pipeline")
        
        # Load data
        df = self.load_data(input_file)
        if df.empty:
            return df
        
        # Apply processing steps
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.standardize_formats(df)
        df = self.synchronize_timestamps(df)
        df = self.compute_midprice(df)
        df = self.compute_future_returns(df)
        
        # Save processed data
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Processed data saved to {output_file}")
        
        logger.info(f"Processing complete. Final dataset has {len(df)} rows")
        return df

def main():
    """Main entry point for data processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Order Book Data Processing Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    
    args = parser.parse_args()
    
    processor = OrderBookDataProcessor(symbol=args.symbol)
    processor.process(input_file=args.input, output_file=args.output)

if __name__ == "__main__":
    main()