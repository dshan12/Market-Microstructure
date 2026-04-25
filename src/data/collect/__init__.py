"""
Data collection module for Binance WebSocket API integration.

Implements Phase 1 of the research pipeline: real-time data collection from
Binance spot market (BTCUSDT, ETHUSDT) with nanosecond/millisecond precision.
"""

from .collector import OrderBookCollector

__all__ = ["OrderBookCollector"]