"""
Market Microstructure Research Project - Core Package

This package contains modules for quantitative analysis of limit order book dynamics
and order flow impact on price discovery.
"""

__version__ = "1.0.0"
__author__ = "Market Microstructure Research Team"
__email__ = "research@example.com"

from .data.collect import OrderBookCollector
from .data.process import OrderBookDataProcessor
from .analysis.run import MarketMicrostructureResearch
from .analysis.run.features import FeatureEngineer
from .analysis.run.statistical import StatisticalAnalyzer

__all__ = [
    "MarketMicrostructureResearch",
    "BinanceOrderBookCollector", 
    "OrderBookDataProcessor",
    "FeatureEngineer",
    "StatisticalAnalyzer"
]