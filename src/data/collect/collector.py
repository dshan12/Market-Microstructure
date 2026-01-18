"""
Binance WebSocket API connector for limit order book data collection.
Phase 1 implementation for quantitative market microstructure research.
"""

import asyncio
import websockets
import json
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BinanceOrderBookCollector:
    """Collect limit order book data from Binance WebSocket API."""
    
    def __init__(self, symbol: str = "BTCUSDT", precision: str = "MILLISECOND"):
        self.symbol = symbol
        self.precision = precision
        self.ws_url = "wss://stream.binance.com:9443/ws/"
        
    async def connect_and_collect(self, duration_seconds: int = 3600):
        """Connect to Binance WebSocket and collect order book data."""
        start_time = datetime.now()
        end_time = start_time + pd.Timedelta(seconds=duration_seconds)
        
        snapshots = []
        
        async with websockets.connect(
            f"{self.ws_url}{self.symbol}@depth"
        ) as websocket:
            logger.info(f"Connected to {self.symbol} order book stream")
            
            while datetime.now() < end_time:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if 'bids' in data and 'asks' in data:
                        timestamp = int(data.get('E', data.get('e', int(datetime.now().timestamp() * 1000))))
                        
                        snapshot = {
                            'timestamp': timestamp,
                            'symbol': self.symbol,
                            'best_bid': float(data['bids'][0][0]) if data['bids'] else None,
                            'best_ask': float(data['asks'][0][0]) if data['asks'] else None,
                            'bid_volume': float(data['bids'][0][1]) if data['bids'] else None,
                            'ask_volume': float(data['asks'][0][1]) if data['asks'] else None,
                            'data': data
                        }
                        
                        snapshots.append(snapshot)
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        logger.info(f"Collected {len(snapshots)} order book snapshots")
        return snapshots
        
    async def run_collection(self, output_file: str = None):
        """Run the complete data collection process."""
        logger.info(f"Starting data collection for {self.symbol}")
        
        snapshots = await self.connect_and_collect(duration_seconds=60)
        
        if snapshots:
            df = pd.DataFrame(snapshots)
            logger.info(f"Collected {len(df)} data points")
            
            if output_file:
                df.to_csv(output_file, index=False)
                logger.info(f"Data saved to {output_file}")
            
            return df
        else:
            logger.warning("No data collected")
            return pd.DataFrame()

def main():
    """Main entry point for data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Binance Order Book Data Collector")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument("--duration", type=int, default=60, help="Collection duration in seconds (default: 60)")
    
    args = parser.parse_args()
    
    collector = BinanceOrderBookCollector(symbol=args.symbol)
    
    async def run():
        await collector.run_collection(output_file=args.output)
    
    asyncio.run(run())

if __name__ == "__main__":
    main()