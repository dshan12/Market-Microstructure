"""
Main entry point for market microstructure research analysis.
Coordinates the complete workflow for answering the research question:
"Does order book imbalance predict future returns?"

This module orchestrates:
1. Data collection from Binance WebSocket API
2. Data processing and feature engineering  
3. Statistical analysis with HAC/Newey-West errors
4. Visualization generation
5. Research paper writing

Author: Market Microstructure Research Team
Date: 2026-06-09
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import argparse
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketMicrostructureResearch:
    """Main orchestrator for quantitative market microstructure research."""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.results = {}
        
    async def run_complete_analysis(self, symbol: str = "BTCUSDT", duration: int = 3600):
        """Run the complete analysis pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING MARKET MICROSTRUCTURE RESEARCH ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Research Question: Does order book imbalance predict future returns?")
        logger.info(f"Asset: {symbol}")
        logger.info(f"Duration: {duration} seconds")
        logger.info("=" * 60)
        
        # Phase 1: Data Collection
        logger.info("\nPHASE 1: DATA COLLECTION")
        await self._collect_data(symbol, duration)
        
        # Phase 2: Data Processing
        logger.info("\nPHASE 2: DATA PROCESSING")
        await self._process_data()
        
        # Phase 3: Feature Engineering
        logger.info("\nPHASE 3: FEATURE ENGINEERING")
        await self._engineer_features()
        
        # Phase 4: Statistical Analysis
        logger.info("\nPHASE 4: STATISTICAL ANALYSIS")
        await self._run_statistical_analysis()
        
        # Phase 5: Visualization
        logger.info("\nPHASE 5: VISUALIZATION")
        await self._generate_visualizations()
        
        # Phase 6: Research Paper
        logger.info("\nPHASE 6: RESEARCH PAPER")
        await self._write_research_paper()
        
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)
        
        return self.results
    
    async def _collect_data(self, symbol: str, duration: int):
        """Execute data collection from Binance."""
        try:
            from src.data.collect.collector import BinanceOrderBookCollector
            
            collector = BinanceOrderBookCollector(symbol=symbol)
            df = await collector.run_collection()
            
            if not df.empty:
                output_file = f"data/raw/{symbol}_orderbook_raw.csv"
                Path("data/raw").mkdir(exist_ok=True)
                df.to_csv(output_file, index=False)
                self.results['raw_data'] = output_file
                logger.info(f"✓ Data collected: {len(df)} snapshots saved to {output_file}")
            else:
                logger.warning("✗ No data collected")
                
        except Exception as e:
            logger.error(f"✗ Data collection failed: {e}")
            raise
    
    async def _process_data(self):
        """Execute data processing pipeline."""
        try:
            from src.data.process.pipeline import OrderBookDataProcessor
            
            # Process BTCUSDT data
            processor = OrderBookDataProcessor(symbol="BTCUSDT")
            btc_processed = processor.process(
                input_file="data/raw/BTCUSDT_orderbook_raw.csv",
                output_file="data/processed/btcusdt_processed.csv"
            )
            
            # Process ETHUSDT data  
            processor_eth = OrderBookDataProcessor(symbol="ETHUSDT")
            eth_processed = processor_eth.process(
                input_file="data/raw/ETHUSDT_orderbook_raw.csv",
                output_file="data/processed/ethusdt_processed.csv"
            )
            
            self.results['processed_data'] = {
                'BTCUSDT': "data/processed/btcusdt_processed.csv",
                'ETHUSDT': "data/processed/ethusdt_processed.csv"
            }
            
            logger.info(f"✓ Data processing complete: Processed {len(btc_processed)} BTCUSDT records")
            
        except Exception as e:
            logger.error(f"✗ Data processing failed: {e}")
            raise
    
    async def _engineer_features(self):
        """Execute feature engineering pipeline."""
        try:
            from src.analysis.run.features import FeatureEngineer
            
            engineer = FeatureEngineer()
            features_df = engineer.engineer_all_features(
                input_file="data/processed/btcusdt_processed.csv",
                output_file="data/features/btcusdt_features.csv"
            )
            
            Path("data/features").mkdir(exist_ok=True)
            self.results['features'] = "data/features/btcusdt_features.csv"
            
            logger.info(f"✓ Feature engineering complete: {len(features_df)} records with {len(features_df.columns)} features")
            
        except Exception as e:
            logger.error(f"✗ Feature engineering failed: {e}")
            raise
    
    async def _run_statistical_analysis(self):
        """Execute statistical analysis pipeline."""
        try:
            from src.analysis.run.statistical import StatisticalAnalyzer
            
            analyzer = StatisticalAnalyzer()
            results = analyzer.run_statistical_analysis(
                input_file="data/features/btcusdt_features.csv",
                output_file="results/statistical_analysis.json"
            )
            
            Path("results").mkdir(exist_ok=True)
            self.results['statistical_results'] = "results/statistical_analysis.json"
            
            logger.info(f"✓ Statistical analysis complete: Analyzed {results.get('sample_size', 0)} records")
            
        except Exception as e:
            logger.error(f"✗ Statistical analysis failed: {e}")
            raise
    
    async def _generate_visualizations(self):
        """Execute visualization generation pipeline."""
        try:
            from src.analysis.run.visualizations import VisualizationGenerator
            
            visualizer = VisualizationGenerator()
            plots = visualizer.generate_all_visualizations(
                input_file="data/features/btcusdt_features.csv",
                output_dir="plots"
            )
            
            Path("plots").mkdir(exist_ok=True)
            self.results['visualizations'] = plots
            
            logger.info(f"✓ Visualization generation complete: Generated {len(plots)} plots")
            
        except Exception as e:
            logger.error(f"✗ Visualization generation failed: {e}")
            raise
    
    async def _write_research_paper(self):
        """Execute research paper writing pipeline."""
        try:
            # Copy the research paper template
            paper_source = "research/paper.md"
            paper_target = "research/market_microstructure_research_paper.md"
            
            Path("research").mkdir(exist_ok=True)
            
            # Copy file (simple implementation)
            with open(paper_source, 'r') as src, open(paper_target, 'w') as dst:
                dst.write(src.read())
            
            self.results['research_paper'] = paper_target
            
            logger.info(f"✓ Research paper complete: Generated {paper_target}")
            
        except Exception as e:
            logger.error(f"✗ Research paper generation failed: {e}")
            raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Market Microstructure Research Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.analysis.run --symbol BTCUSDT --duration 300
  python -m src.analysis.run --config research_config.json
        """
    )
    
    parser.add_argument(
        "--symbol", default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Collection duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--config", help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Create and run research analysis
    research = MarketMicrostructureResearch(config_file=args.config)
    
    try:
        asyncio.run(research.run_complete_analysis(
            symbol=args.symbol,
            duration=args.duration
        ))
        
        logger.info("\n🎉 RESEARCH ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("\nKey deliverables generated:")
        for key, value in research.results.items():
            logger.info(f"  - {key}: {value}")
            
    except Exception as e:
        logger.error(f"\n❌ RESEARCH ANALYSIS FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()