"""
Exploratory data analysis visualizations.
Phase 4 implementation for quantitative market microstructure research.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Generate EDA visualizations for limit order book analysis."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        plt.style.use('seaborn-v0_8')
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data for visualization."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} data points for visualization")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def plot_distribution(self, df: pd.DataFrame, column: str, title: str, bins: int = 50):
        """Create histogram of distribution for a given column."""
        if df.empty or column not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(column.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        
        return fig
    
    def plot_time_series(self, df: pd.DataFrame, value_cols: list, timestamp_col: str = 'timestamp', title: str = "Time Series"):
        """Create time series plot for specified columns."""
        if df.empty:
            return None
        
        # Convert timestamp to datetime if needed
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df.set_index(timestamp_col, inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in value_cols:
            if col in df.columns:
                ax.plot(df.index, df[col], label=col, linewidth=1)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_heatmap(self, df: pd.DataFrame, x_col: str, y_col: str, title: str = "Heatmap"):
        """Create heatmap of relationship between two columns."""
        if df.empty:
            return None
        
        # Create a pivot table for heatmap
        pivot_df = df.pivot_table(values=y_col, columns=x_col, aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap='RdBu_r', center=0, ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig
    
    def plot_decile_analysis(self, df: pd.DataFrame, imbalance_col: str = 'depth_imbalance', return_col: str = 'return_1s'):
        """Create decile analysis plot showing imbalance vs future returns."""
        if df.empty:
            return None
        
        # Create deciles based on imbalance
        df_sorted = df.sort_values(by=imbalance_col)
        df_sorted['imbalance_decile'] = pd.qcut(df_sorted[imbalance_col], 10)
        
        # Calculate average returns per decile
        decile_stats = df_sorted.groupby('imbalance_decile')[return_col].agg(['mean', 'std']).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Distribution of imbalance by decile
        ax1.bar(range(1, 11), decile_stats[imbalance_col].astype(str), color='lightblue')
        ax1.set_xlabel('Imbalance Decile')
        ax1.set_ylabel('Imbalance Range')
        ax1.set_title('Imbalance Distribution by Decile', fontweight='bold')
        
        # Plot 2: Average returns by decile
        colors = ['red' if x < 0 else 'green' for x in decile_stats['mean']]
        bars = ax2.bar(range(1, 11), decile_stats['mean'], color=colors)
        ax2.set_xlabel('Imbalance Decile')
        ax2.set_ylabel('Average Future Return')
        ax2.set_title('Average Future Returns by Imbalance Decile', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add error bars
        ax2.errorbar(range(1, 11), decile_stats['mean'], 
                    yerr=decile_stats['std'], fmt='none', color='black', capsize=5)
        
        plt.tight_layout()
        return fig
    
    def generate_all_visualizations(self, input_file: str, output_dir: str = "plots"):
        """Generate all EDA visualizations."""
        logger.info("Starting visualization generation pipeline")
        
        # Load data
        df = self.load_data(input_file)
        if df.empty:
            return
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        visualizations = {}
        
        # 1. Distribution of imbalance
        if 'depth_imbalance' in df.columns:
            fig = self.plot_distribution(
                df, 'depth_imbalance', 
                "Distribution of Depth Imbalance", 
                bins=50
            )
            if fig:
                fig.savefig(f"{output_dir}/imbalance_distribution.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['imbalance_distribution'] = f"{output_dir}/imbalance_distribution.png"
                logger.info("Saved imbalance distribution plot")
        
        # 2. Distribution of spread
        if 'spread' in df.columns:
            fig = self.plot_distribution(
                df, 'spread', 
                "Distribution of Spread", 
                bins=50
            )
            if fig:
                fig.savefig(f"{output_dir}/spread_distribution.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['spread_distribution'] = f"{output_dir}/spread_distribution.png"
                logger.info("Saved spread distribution plot")
        
        # 3. Time series of liquidity
        time_series_cols = ['depth_imbalance', 'spread', 'midprice']
        fig = self.plot_time_series(
            df, time_series_cols, 
            "Time Series of Liquidity Metrics",
            "Liquidity Metrics Time Series"
        )
        if fig:
            fig.savefig(f"{output_dir}/liquidity_time_series.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            visualizations['liquidity_time_series'] = f"{output_dir}/liquidity_time_series.png"
            logger.info("Saved liquidity time series plot")
        
        # 4. Heatmap of imbalance vs returns
        if 'depth_imbalance' in df.columns:
            # Sample data for heatmap (take every 100th row)
            sample_df = df.iloc[::100].copy()
            fig = self.plot_heatmap(
                sample_df, 'depth_imbalance', 'return_1s',
                "Depth Imbalance vs 1-Second Future Return"
            )
            if fig:
                fig.savefig(f"{output_dir}/imbalance_return_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['imbalance_return_heatmap'] = f"{output_dir}/imbalance_return_heatmap.png"
                logger.info("Saved imbalance-return heatmap")
        
        # 5. Decile analysis
        fig = self.plot_decile_analysis(df)
        if fig:
            fig.savefig(f"{output_dir}/decile_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            visualizations['decile_analysis'] = f"{output_dir}/decile_analysis.png"
            logger.info("Saved decile analysis plot")
        
        logger.info(f"Generated {len(visualizations)} visualizations")
        return visualizations

def main():
    """Main entry point for visualization generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization Generation Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default="plots", help="Output directory for plots (default: plots)")
    
    args = parser.parse_args()
    
    visualizer = VisualizationGenerator(output_dir=args.output)
    visualizer.generate_all_visualizations(input_file=args.input, output_dir=args.output)

if __name__ == "__main__":
    main()