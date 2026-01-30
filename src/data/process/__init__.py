"""
Data processing pipeline for quantitative market microstructure research.

Implements Phase 2 of the research pipeline: conversion of raw exchange data
into research-ready format with comprehensive feature engineering.
"""

from .pipeline import OrderBookDataProcessor

__all__ = ["OrderBookDataProcessor"]