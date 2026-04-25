"""
Real-time order book data collection via WebSocket.

Connects to Kraken (ws.kraken.com) ticker feed — accessible from US,
unlike Binance which geo-blocks.  Collects best bid/ask with volume
every time the market updates (~100-200ms for BTC/USD).
"""

import asyncio
import websockets
import json
from datetime import datetime
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class OrderBookCollector:
    """Collect top-of-book ticker data from Kraken WebSocket."""

    def __init__(self, symbol: str = "XBT/USD"):
        self.symbol = symbol
        self.pair = symbol
        self.url = "wss://ws.kraken.com"

    async def collect(self, duration_seconds: int = 600) -> pd.DataFrame:
        """Connect to Kraken and collect ticker data."""
        start_time = datetime.now()
        end_time = start_time + pd.Timedelta(seconds=duration_seconds)
        rows = []

        subscribe = {
            "event": "subscribe",
            "pair": [self.pair],
            "subscription": {"name": "ticker"},
        }

        async with websockets.connect(self.url) as ws:
            # Send subscribe
            await ws.send(json.dumps(subscribe))
            # Read subscription confirmation
            conf = await ws.recv()
            logger.info(f"Subscribed: {json.loads(conf)}")

            while datetime.now() < end_time:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(msg)
                    # Kraken ticker data comes as a list:
                    #   [channel_id, {ticker_dict}, "channel_name", "pair"]
                    if isinstance(data, list) and len(data) >= 2:
                        t = data[1]
                        if not isinstance(t, dict) or 'b' not in t:
                            continue
                        ts_ms = int(datetime.now().timestamp() * 1000)
                        rows.append({
                            'timestamp': ts_ms,
                            'best_bid': float(t['b'][0]),
                            'best_ask': float(t['a'][0]),
                            'bid_volume': float(t['b'][2]),
                            'ask_volume': float(t['a'][2]),
                        })
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        df = pd.DataFrame(rows)
        logger.info(f"Collected {len(df)} ticker snapshots over {duration_seconds}s")
        return df

    async def run_collection(self, output_file: Optional[str] = None,
                             duration_seconds: int = 600) -> pd.DataFrame:
        logger.info(f"Starting data collection for {self.symbol} ({duration_seconds}s)")
        df = await self.collect(duration_seconds=duration_seconds)
        if df.empty:
            logger.warning("No data collected")
            return df
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Data saved to {output_file}")
        return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Order Book Data Collector")
    parser.add_argument("--symbol", default="XBT/USD", help="Trading pair (default: XBT/USD)")
    parser.add_argument("--output", default="data/raw/kraken_ticker.csv",
                        help="Output CSV file path")
    parser.add_argument("--duration", type=int, default=600,
                        help="Collection duration in seconds (default: 600 = 10 min)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    collector = OrderBookCollector(symbol=args.symbol)

    async def run():
        await collector.run_collection(
            output_file=args.output, duration_seconds=args.duration
        )

    asyncio.run(run())


if __name__ == "__main__":
    main()
